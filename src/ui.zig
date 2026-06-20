const c = @import("c.zig").c;
const state = @import("state.zig");
const utils = @import("utils.zig");
const std = @import("std");
const scene = @import("scene.zig");

// TODO: maybe put frag path in appstate idk man ill figure it out
// light temperature is also fucked gg, will fix later:tm:
pub fn render(appState: *state.app_state) !void {
    // const imgui_io = c.ImGui_GetIO();
    c.cImGui_ImplOpenGL3_NewFrame();
    c.cImGui_ImplGlfw_NewFrame();
    c.ImGui_NewFrame();

    c.ImGui_SetNextWindowSize(appState.imgui.window_size, c.ImGuiCond_FirstUseEver);
    c.ImGui_SetNextWindowPos(appState.imgui.window_pos, c.ImGuiCond_FirstUseEver);

    _ = c.ImGui_Begin("Demos", &appState.imgui.is_active, c.ImGuiWindowFlags_NoSavedSettings | c.ImGuiWindowFlags_NoTitleBar);
    if (c.ImGui_Button("Save Render")) {
        appState.renderer.want_to_save = true;
        var w: c_int = 0;
        var h: c_int = 0;
        c.glfwGetFramebufferSize(appState.window, &w, &h);

        // Note: In Zig, error handling in UI callbacks can be tricky.
        // We use 'catch' to prevent a crash if the disk is full.
        // utils.saveScreenshot(allocator, w, h, appState.renderer.total_accumulated_frames) catch |err| {
        //     std.log.err("Failed to save screenshot: {}", .{err});
        // };
    }
    const current_idx = @as(usize, @intCast(appState.scenes.requested_scene_index));
    const scene_names = appState.scenes.registry;
    const preview_name = appState.scenes.registry[current_idx].metadata.name.ptr;
    if (c.ImGui_BeginCombo(" ", preview_name, 0)) {
        for (scene_names, 0..) |name, i| {
            if (c.ImGui_Selectable(name.metadata.name)) {
                appState.scenes.requested_scene_index = @intCast(i);
                appState.renderer.total_accumulated_frames = 0;
            }
        }
        c.ImGui_EndCombo();
    }
    c.ImGui_Spacing();

    c.ImGui_SeparatorText("Metrics");
    c.ImGui_Text("Frame: %d", appState.renderer.total_accumulated_frames);
    c.ImGui_Text("Frame time: %.2f ms", appState.metrics.ms_per_frame);

    c.ImGui_SeparatorText("Camera Control");
    if (c.ImGui_SliderFloat("Aperture", @ptrCast(&appState.scenes.current_state.aperture), 0.0, 2.0)) {
        // std.debug.print("aperture:{d}\n", .{appState.scenes.current_state.aperture});
        appState.renderer.should_reset_accumulation = true;
    }
    if (c.ImGui_SliderFloat("Focal Distance", @ptrCast(&appState.scenes.current_state.focal_distance), 0.5, 100)) {
        // std.debug.print("focal distance:{d}\n", .{appState.scenes.current_state.focal_distance});
        appState.renderer.should_reset_accumulation = true;
    }

    c.ImGui_SeparatorText("Position");
    c.ImGui_Text("x: %.2f", appState.scenes.current_state.cam_pos[0]);
    c.ImGui_Text("y: %.2f", appState.scenes.current_state.cam_pos[1]);
    c.ImGui_Text("z: %.2f", appState.scenes.current_state.cam_pos[2]);
    c.ImGui_Text("yaw: %.2f", appState.scenes.current_state.yaw);
    c.ImGui_Text("pitch: %.2f", appState.scenes.current_state.pitch);

    c.ImGui_Spacing();
    c.ImGui_Spacing();
    // if (c.ImGui_SliderInt("Color temperature", @ptrCast(&light_temperature), 1000, 7000)) {
    //     // std.debug.print("temp:{any}\n", .{light_temperature});
    // }

    c.ImGui_SeparatorText("Animation");
    const active_entry = appState.scenes.registry[appState.scenes.active_index];
    if (active_entry.animate_fn == null) {
        c.ImGui_Text("No animation for this scene");
    } else if (appState.renderer.anim_rendering) {
        // A render is in flight: only show progress + cancel.
        c.ImGui_Text("Rendering frame %d / %d", appState.renderer.anim_frame_index, appState.renderer.anim_total_frames);
        if (c.ImGui_Button("Cancel Render")) {
            appState.renderer.want_cancel_render = true;
        }
    } else {
        if (appState.renderer.anim_playing) {
            if (c.ImGui_Button("Stop Animation")) {
                appState.renderer.anim_playing = false;
            }
        } else {
            if (c.ImGui_Button("Play Animation")) {
                appState.renderer.anim_playing = true;
                appState.renderer.anim_time = 0;
                appState.renderer.should_reset_accumulation = true;
            }
        }

        if (c.ImGui_Button("Render Animation")) {
            appState.renderer.want_to_render = true;
        }

        _ = c.ImGui_SliderFloat("FPS", @ptrCast(&appState.renderer.anim_fps), 1.0, 60.0);

        var spf_anim: c_int = @intCast(appState.renderer.samples_per_anim_frame);
        if (c.ImGui_SliderInt("Samples/frame", @ptrCast(&spf_anim), 1, 4096)) {
            appState.renderer.samples_per_anim_frame = @intCast(spf_anim);
        }

        _ = c.ImGui_SliderInt("CRF (quality)", @ptrCast(&appState.renderer.render_crf), 0, 51);

        c.ImGui_Text("Time: %.2f / %.2f s", appState.renderer.anim_time, active_entry.anim_duration);
    }

    c.ImGui_End();

    c.ImGui_Render();
}
