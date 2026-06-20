const gl = @import("gl");
const shaders = @import("shaders.zig");
const std = @import("std");
const zstbi = @import("zstbi");
const utils = @import("utils.zig");
const state = @import("state.zig");
const cli = @import("cli.zig");
const c = @import("c.zig").c;
const ui = @import("ui.zig");
const scenes = @import("scenes/scenes.zig");

fn errorCallback(errn: c_int, str: [*c]const u8) callconv(std.builtin.CallingConvention.c) void {
    std.log.err("GLFW Error '{}'': {s}", .{ errn, str });
}

fn cursorPosCallback(window: ?*c.GLFWwindow, xpos: f64, ypos: f64) callconv(std.builtin.CallingConvention.c) void {
    const user_ptr = c.glfwGetWindowUserPointer(window);
    const app = @as(*state.app_state, @ptrCast(@alignCast(user_ptr)));

    if (!app.mouse.is_captured) return;

    if (app.mouse.first_mouse) {
        app.mouse.last_pos[0] = xpos;
        app.mouse.last_pos[1] = ypos;
        app.mouse.first_mouse = false;
        return;
    }

    const x_offset = xpos - app.mouse.last_pos[0];
    const y_offset = app.mouse.last_pos[1] - ypos; // Y is inverted in screen space

    app.mouse.last_pos[0] = xpos;
    app.mouse.last_pos[1] = ypos;

    const sensitivity: f32 = 0.005;
    app.scenes.current_state.yaw -= @as(f32, @floatCast(x_offset)) * sensitivity;
    app.scenes.current_state.pitch -= @as(f32, @floatCast(y_offset)) * sensitivity;

    // Constrain pitch (approx 85 degrees)
    if (app.scenes.current_state.pitch > 1.5) app.scenes.current_state.pitch = 1.5;
    if (app.scenes.current_state.pitch < -1.5) app.scenes.current_state.pitch = -1.5;

    // This flag tells your main loop to reset u_frame and clear the ping-pong buffers
    app.mouse.moved = true;
    app.renderer.should_reset_accumulation = true;
}

fn mouseButtonCallback(window: ?*c.GLFWwindow, button: c_int, action: c_int, mods: c_int) callconv(std.builtin.CallingConvention.c) void {
    _ = mods;
    const user_ptr = c.glfwGetWindowUserPointer(window);
    const app = @as(*state.app_state, @ptrCast(@alignCast(user_ptr)));

    const imgui_io = c.ImGui_GetIO();
    app.mouse.imgui_wants_mouse = imgui_io.*.WantCaptureMouse;

    if (app.mouse.imgui_wants_mouse and action == c.GLFW_PRESS) {
        return;
    }

    if (button == c.GLFW_MOUSE_BUTTON_LEFT) {
        if (action == c.GLFW_PRESS) {
            app.mouse.is_captured = true;
            app.mouse.lmb_pressed = true;
            c.glfwSetInputMode(window, c.GLFW_CURSOR, c.GLFW_CURSOR_DISABLED);
            std.debug.print("pressed lmb\n", .{});

            app.mouse.first_mouse = true;
        } else if (action == c.GLFW_RELEASE) {
            app.mouse.is_captured = false;
            app.mouse.lmb_pressed = false; // Resetting the pressed state
            c.glfwSetInputMode(window, c.GLFW_CURSOR, c.GLFW_CURSOR_NORMAL);
        }
    }
}

pub fn main() !void {
    var procs: gl.ProcTable = undefined;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    if (c.glfwInit() != c.GLFW_TRUE) {
        return;
    }
    defer c.glfwTerminate();

    var appState = state.app_state{
        .scenes = .{
            .allocator = allocator,
            .registry = scenes.registry,
        },
    };
    appState.window_title = "SDF Path Tracing";

    // CLI ARGS
    try cli.processArgs(&appState, allocator);

    const GLSL_VERSION = "#version 460";

    c.glfwWindowHint(c.GLFW_CONTEXT_VERSION_MAJOR, 4);
    c.glfwWindowHint(c.GLFW_CONTEXT_VERSION_MINOR, 6);
    c.glfwWindowHint(c.GLFW_OPENGL_PROFILE, c.GLFW_OPENGL_CORE_PROFILE);
    appState.window = c.glfwCreateWindow(appState.window_w, appState.window_h, appState.window_title.ptr, null, null);
    if (appState.window == null) {
        return;
    }
    defer c.glfwDestroyWindow(appState.window);

    c.glfwSetWindowUserPointer(appState.window, &appState);

    // Callbacks
    _ = c.glfwSetCursorPosCallback(appState.window, cursorPosCallback);
    _ = c.glfwSetErrorCallback(errorCallback);
    _ = c.glfwSetMouseButtonCallback(appState.window, mouseButtonCallback);

    c.glfwMakeContextCurrent(appState.window);
    c.glfwSwapInterval(1);

    if (!procs.init(c.glfwGetProcAddress)) return error.InitFailed;

    gl.makeProcTableCurrent(&procs);
    defer gl.makeProcTableCurrent(null);

    const version_ptr = gl.GetString(gl.VERSION);
    const version_string = std.mem.span(version_ptr);
    std.debug.print("OpenGL Version:{s}\n", .{version_string orelse "Unknown"});

    const EBO = try utils.fullscreenQuad();

    // Textures
    zstbi.init(allocator);
    defer zstbi.deinit();

    try appState.scenes.switchScene(0);

    // main (tone mapping pass)
    const vert_source = try shaders.readFileToString(allocator, "shaders/shader.vert");
    defer allocator.free(vert_source);
    const vert = try shaders.compileShader(allocator, vert_source, gl.VERTEX_SHADER);
    defer allocator.free(vert_source);
    const main_source = try shaders.readFileToString(allocator, "shaders/main.frag");
    const mainFS = try shaders.compileShader(allocator, main_source, gl.FRAGMENT_SHADER);
    const mainProg = try shaders.setupShaderProgram(allocator, &[_]u32{ mainFS, vert });
    defer allocator.free(main_source);

    //FRAME BUFFERS
    var fbos: [2]u32 = undefined;
    var pass1_textures: [2]u32 = undefined;
    var current: u32 = 0;

    gl.GenFramebuffers(2, @ptrCast(&fbos));
    gl.GenTextures(2, @ptrCast(&pass1_textures));
    for (0..2) |i| {
        //texture we write to
        gl.BindTexture(gl.TEXTURE_2D, pass1_textures[i]);
        gl.TexImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, appState.renderer.render_w, appState.renderer.render_h, 0, gl.RGBA, gl.FLOAT, null);
        gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

        //attach texture to framebuffer
        gl.BindFramebuffer(gl.FRAMEBUFFER, fbos[i]);
        gl.FramebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, pass1_textures[i], 0);
    }
    //cleanup
    gl.BindFramebuffer(gl.FRAMEBUFFER, 0);
    gl.BindTexture(gl.TEXTURE_2D, 0);

    // Dedicated FBO for screenshots at render resolution (so saved PNGs match
    // render_w/render_h, not the window size). The tone-map pass renders into
    // this on demand and we ReadPixels from it.
    var screenshot_fbo: u32 = undefined;
    var screenshot_texture: u32 = undefined;
    gl.GenFramebuffers(1, @ptrCast(&screenshot_fbo));
    gl.GenTextures(1, @ptrCast(&screenshot_texture));
    gl.BindTexture(gl.TEXTURE_2D, screenshot_texture);
    gl.TexImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, appState.renderer.render_w, appState.renderer.render_h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.BindFramebuffer(gl.FRAMEBUFFER, screenshot_fbo);
    gl.FramebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, screenshot_texture, 0);
    gl.BindFramebuffer(gl.FRAMEBUFFER, 0);
    gl.BindTexture(gl.TEXTURE_2D, 0);

    _ = c.CIMGUI_CHECKVERSION();
    _ = c.ImGui_CreateContext(null);
    defer c.ImGui_DestroyContext(null);

    const imio = c.ImGui_GetIO();
    imio.*.ConfigFlags = c.ImGuiConfigFlags_NavEnableKeyboard;

    c.ImGui_StyleColorsDark(null);

    _ = c.cImGui_ImplGlfw_InitForOpenGL(appState.window, true);
    defer c.cImGui_ImplGlfw_Shutdown();

    _ = c.cImGui_ImplOpenGL3_InitEx(GLSL_VERSION);
    defer c.cImGui_ImplOpenGL3_Shutdown();

    // TODO:
    // figure out what to do this these variables
    // probably want to remove rendered frame on the
    // screenshot and video rework
    // spf_frames can maybe go to state
    var spf_frames: u32 = 0;
    var rendered_frame: u32 = 1;

    appState.imgui.is_active = false;
    appState.scenes.requested_scene_index = 0;
    appState.imgui.window_size = .{ .x = 0, .y = 0 };
    appState.renderer.total_accumulated_frames = 0;
    appState.metrics.ms_per_frame = 0;
    appState.renderer.want_to_save = false;
    appState.scenes.requested_scene_index = 0;

    //scene state
    // FIX: want to move this inside the scene, so each scene can define their own kind of these things
    const light_temperature: i32 = 5000;

    appState.metrics.last_time = c.glfwGetTime();
    while (c.glfwWindowShouldClose(appState.window) == 0) {
        c.glfwPollEvents();

        if (appState.scenes.requested_scene_index != appState.scenes.active_index) {
            try appState.scenes.switchScene(@intCast(appState.scenes.requested_scene_index));

            appState.renderer.should_reset_accumulation = true;
            appState.renderer.total_accumulated_frames = 0;
        }

        const current_scene = appState.scenes.active_scene;

        // UI
        try ui.render(&appState);

        // TODO: need to take a look at these variables
        var width: c_int = 0;
        var height: c_int = 0;
        c.glfwGetFramebufferSize(appState.window, &width, &height);
        gl.Viewport(0, 0, appState.renderer.render_w, appState.renderer.render_h);
        gl.Clear(gl.COLOR_BUFFER_BIT);

        //uniforms setup

        gl.UseProgram(current_scene.?.shader_program);
        var mouse_x: f64 = undefined;
        var mouse_y: f64 = undefined;
        c.glfwGetCursorPos(appState.window, @ptrCast(&mouse_x), @ptrCast(&mouse_y));

        const uniform_window_size = gl.GetUniformLocation(current_scene.?.shader_program, "u_resolution");
        // gl.Uniform2f(uniform_window_size, @floatFromInt(width), @floatFromInt(height));
        gl.Uniform2f(uniform_window_size, @floatFromInt(appState.renderer.render_w), @floatFromInt(appState.renderer.render_h));

        const uniform_mouse_pos = gl.GetUniformLocation(current_scene.?.shader_program, "u_mouse");
        gl.Uniform2f(uniform_mouse_pos, @floatCast(mouse_x), @floatCast(mouse_y));

        const current_time: f64 = c.glfwGetTime();
        const uniform_time = gl.GetUniformLocation(current_scene.?.shader_program, "u_time");
        gl.Uniform1f(uniform_time, @floatCast(current_time));

        const uniform_frame = gl.GetUniformLocation(current_scene.?.shader_program, "u_frame");
        gl.Uniform1i(uniform_frame, @intCast(appState.renderer.total_accumulated_frames));

        gl.Uniform1i(gl.GetUniformLocation(current_scene.?.shader_program, "u_spf"), appState.renderer.samples_per_frame);

        const temperatureRGB = utils.kelvinToColor(light_temperature);
        const uniform_temperatureRGB = gl.GetUniformLocation(current_scene.?.shader_program, "u_tempColor");
        gl.Uniform3f(uniform_temperatureRGB, temperatureRGB[0], temperatureRGB[1], temperatureRGB[2]);

        // yaw and pitch
        const uniform_cam_rot = gl.GetUniformLocation(current_scene.?.shader_program, "u_cameraRot");
        gl.Uniform2f(uniform_cam_rot, appState.scenes.current_state.yaw, appState.scenes.current_state.pitch);

        //camera position
        const uniform_cam_pos = gl.GetUniformLocation(current_scene.?.shader_program, "u_cameraPos");
        gl.Uniform3fv(uniform_cam_pos, 1, @ptrCast(&appState.scenes.current_state.cam_pos));

        // aperture
        const uniform_aperture = gl.GetUniformLocation(current_scene.?.shader_program, "u_aperture");
        gl.Uniform1f(uniform_aperture, @floatCast(appState.scenes.current_state.aperture));
        // focal distance
        const unifrom_focal_distance = gl.GetUniformLocation(current_scene.?.shader_program, "u_focusDist");
        gl.Uniform1f(unifrom_focal_distance, @floatCast(appState.scenes.current_state.focal_distance));

        if (utils.handleMovement(&appState)) {
            appState.renderer.should_reset_accumulation = true;
        }

        appState.scenes.current_state.bound_texture_count = 0;
        for (current_scene.?.textures, current_scene.?.texture_names) |texture, name| {
            try utils.bindTexture(texture, name, current_scene.?.shader_program, &appState);
        }

        //drawing fbo
        gl.ActiveTexture(gl.TEXTURE0);
        gl.BindTexture(gl.TEXTURE_2D, pass1_textures[current]);
        gl.Uniform1i(gl.GetUniformLocation(current_scene.?.shader_program, "u_pass1"), 0);
        gl.BindBuffer(gl.ELEMENT_ARRAY_BUFFER, EBO);
        gl.BindFramebuffer(gl.FRAMEBUFFER, fbos[1 - current]);
        const imgui_io = c.ImGui_GetIO();
        if ((appState.mouse.moved or appState.renderer.should_reset_accumulation) and imgui_io.*.WantCaptureMouse == false) {
            const clear_color = [4]f32{ 0.0, 0.0, 0.0, 0.0 };
            gl.ClearBufferfv(gl.COLOR, 0, @ptrCast(&clear_color));

            // Temporarily bind and clear source
            gl.BindFramebuffer(gl.FRAMEBUFFER, fbos[current]);
            gl.ClearBufferfv(gl.COLOR, 0, @ptrCast(&clear_color));

            // Re-bind destination
            gl.BindFramebuffer(gl.FRAMEBUFFER, fbos[1 - current]);
            appState.renderer.total_accumulated_frames = 0;
            appState.mouse.moved = false;

            appState.renderer.should_reset_accumulation = false;
        }
        gl.DrawElements(gl.TRIANGLES, 6, gl.UNSIGNED_INT, 0);
        //invert framebuffer
        current = 1 - current;

        // blit to screen
        gl.BindFramebuffer(gl.FRAMEBUFFER, 0);

        var win_w: c_int = 0;
        var win_h: c_int = 0;
        c.glfwGetFramebufferSize(appState.window, &win_w, &win_h);
        gl.Viewport(0, 0, win_w, win_h); // Shrink viewport for the screen

        gl.UseProgram(mainProg);
        gl.ActiveTexture(gl.TEXTURE0);
        gl.BindTexture(gl.TEXTURE_2D, pass1_textures[current]);
        gl.Uniform1i(gl.GetUniformLocation(mainProg, "u_pass1"), 0);
        gl.Uniform2f(gl.GetUniformLocation(mainProg, "u_resolution"), @floatFromInt(width), @floatFromInt(height));
        gl.DrawElements(gl.TRIANGLES, 6, gl.UNSIGNED_INT, 0);

        if (appState.renderer.want_to_save) {
            // Re-run the tone-map pass into the render-resolution screenshot FBO so the
            // saved image matches render_w/render_h (not the window). main.frag computes
            // UVs as fragCoord / u_resolution, so the viewport and u_resolution must both
            // be set to render res for a 1:1 sample of the accumulation texture.
            gl.BindFramebuffer(gl.FRAMEBUFFER, screenshot_fbo);
            gl.Viewport(0, 0, appState.renderer.render_w, appState.renderer.render_h);
            gl.UseProgram(mainProg);
            gl.ActiveTexture(gl.TEXTURE0);
            gl.BindTexture(gl.TEXTURE_2D, pass1_textures[current]);
            gl.Uniform1i(gl.GetUniformLocation(mainProg, "u_pass1"), 0);
            gl.Uniform2f(gl.GetUniformLocation(mainProg, "u_resolution"), @floatFromInt(appState.renderer.render_w), @floatFromInt(appState.renderer.render_h));
            gl.DrawElements(gl.TRIANGLES, 6, gl.UNSIGNED_INT, 0);

            // saveScreenshot reads the currently bound FBO, so keep screenshot_fbo bound.
            utils.saveScreenshot(allocator, appState.renderer.render_w, appState.renderer.render_h, appState.renderer.total_accumulated_frames) catch |err| {
                std.log.err("Failed to save screenshot: {}", .{err});
            };

            // Restore window framebuffer + viewport for the ImGui draw that follows.
            gl.BindFramebuffer(gl.FRAMEBUFFER, 0);
            gl.Viewport(0, 0, win_w, win_h);
            rendered_frame += 1;
            appState.renderer.want_to_save = false;
        }

        while (gl.GetError() != gl.NO_ERROR) {
            std.debug.print("OpenGL Error:{any}\n", .{gl.GetError()});
        }
        c.cImGui_ImplOpenGL3_RenderDrawData(c.ImGui_GetDrawData());

        appState.renderer.total_accumulated_frames += 1;
        spf_frames += 1;
        const spf_current_time = c.glfwGetTime();
        if (spf_current_time - appState.metrics.last_time >= 1.0) {
            appState.metrics.ms_per_frame = 1000 / @as(f32, @floatFromInt(spf_frames));
            spf_frames = 0;
            appState.metrics.last_time += 1.0;
        }
        c.glfwSwapBuffers(appState.window);
    }
}
