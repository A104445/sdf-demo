const std = @import("std");
const math = std.math;
const zstbi = @import("zstbi");
const gl = @import("gl");
const state = @import("state.zig");
const c = @import("c.zig").c;

pub fn kelvinToColor(temp_in: i32) [3]f32 {
    // const temp = temp_in / 100;
    const temp_f: f32 = @floatFromInt(temp_in);
    const temp = temp_f / 100.0;
    var red: f32 = undefined;
    var green: f32 = undefined;
    var blue: f32 = undefined;

    if (temp <= 66) {
        red = 255;
    } else {
        red = temp - 60;
        red = 329.698727446 * (math.pow(f32, red, -0.1332047592));
        red = math.clamp(red, 0, 255);
    }

    if (temp <= 66) {
        green = temp;
        green = 99.4708025861 * @log(green) - 161.1195681661;
    } else {
        green = temp - 60;
        green = 288.1221695283 * (math.pow(f32, green, -0.0755148492));
        green = math.clamp(green, 0, 255);
    }

    if (temp >= 66) {
        blue = 255;
    } else {
        if (temp <= 19) {
            blue = 0;
        } else {
            blue = temp - 10;
            blue = 138.5177312231 * @log(blue) - 305.0447927307;
            blue = math.clamp(blue, 0, 255);
        }
    }

    // std.debug.print("red:{d},green:{d},blue:{d}\n", .{ red, green, blue });

    return [_]f32{ red / 255.0, green / 255.0, blue / 255.0 };
}

pub fn loadTexture(path: [:0]const u8, index: u32) !void {
    std.debug.print("loading asset: {s}\n", .{path});
    var image = try zstbi.Image.loadFromFile(path, 4);
    defer image.deinit();

    gl.BindTexture(gl.TEXTURE_2D, index);
    gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);

    gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    const is_16_bit = image.bytes_per_component == 2;

    const internal_format: i32 = if (is_16_bit) gl.RGBA16 else gl.RGBA;
    const data_type: u32 = if (is_16_bit) gl.UNSIGNED_SHORT else gl.UNSIGNED_BYTE;

    gl.TexImage2D(
        gl.TEXTURE_2D,
        0,
        internal_format,
        @intCast(image.width),
        @intCast(image.height),
        0,
        gl.RGBA,
        data_type,
        image.data.ptr,
    );
    gl.GenerateMipmap(gl.TEXTURE_2D);
}

//temp image saver, ill do my own probably
pub fn saveScreenshot(allocator: std.mem.Allocator, width: c_int, height: c_int, file_n: u32) !void {
    const w = @as(usize, @intCast(width));
    const h = @as(usize, @intCast(height));
    const stride = w * 4; // 3 bytes per pixel (RGB)
    const total_size = stride * h;

    // 1. Allocate a buffer to hold the pixel data
    const pixels = try allocator.alloc(u8, total_size);
    defer allocator.free(pixels);

    // 2. Read the pixels from the currently active OpenGL Framebuffer
    // We use gl.RGB and gl.UNSIGNED_BYTE to get standard 24-bit color
    // gl.PixelStorei(gl.PACK_ALIGNMENT, 1); // Tell GL not to add padding
    // gl.ReadPixels(0, 0, width, height, gl.RGB, gl.UNSIGNED_BYTE, pixels.ptr);
    gl.ReadPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels.ptr);

    // 3. Flip the image vertically
    // OpenGL's (0,0) is bottom-left, but image files expect (0,0) at top-left
    const half_h = h / 2;
    for (0..half_h) |y| {
        const top_idx = y * stride;
        const bot_idx = (h - 1 - y) * stride;

        for (0..stride) |x| {
            const temp = pixels[top_idx + x];
            pixels[top_idx + x] = pixels[bot_idx + x];
            pixels[bot_idx + x] = temp;
        }
    }

    // 4. Write to a PPM file
    const filename: [:0]const u8 = try std.fmt.allocPrintSentinel(allocator, "render/frame-{d:0>3}.png", .{file_n}, 0);
    const file = try std.fs.cwd().createFile(filename, .{});
    defer file.close();

    // 5. Map our raw pixel data into the zig-gamedev zstbi.Image struct␍
    const image = zstbi.Image{
        .data = pixels,
        .width = @as(u32, @intCast(width)),
        .height = @as(u32, @intCast(height)),
        .num_components = 4, // RGBA
        .bytes_per_component = 1, // 8-bit channels
        .bytes_per_row = @as(u32, @intCast(stride)),
        .is_hdr = false,
    };

    // 6. Tell the struct to save itself to disk
    try image.writeToFile(filename, .png);

    std.debug.print("Screenshot saved to {s}!\n", .{filename});
}

// FIX: maybe look at how i should i actually do the shader program part, the scenes array would probably fix this
pub fn bindTexture(texture_id: u32, uniform_name: [:0]const u8, shader_program: u32, appState: *state.app_state) !void {
    const current_max_texture = appState.scenes.current_state.bound_texture_count;

    //gl.TEXTURE0 is reserved for the path tracing pass
    gl.ActiveTexture(@as(u32, gl.TEXTURE1) + current_max_texture);
    gl.BindTexture(gl.TEXTURE_2D, texture_id);
    gl.Uniform1i(gl.GetUniformLocation(shader_program, uniform_name.ptr), @intCast(current_max_texture + 1));
    appState.scenes.current_state.bound_texture_count += 1;
}

pub fn fullscreenQuad() !u32 {
    const vertices = [_]f32{
        -1.0, 1.0, 0.0, //top left
        -1.0, -1.0, 0.0, //bottom left
        1.0, -1.0, 0.0, //bottom right
        1.0, 1.0, 0.0, //top right
    };

    const indices = [_]u32{
        0, 1, 2, // 1
        0, 2, 3, // 2
    };

    var VAO: u32 = undefined;
    gl.GenVertexArrays(1, @ptrCast(&VAO));
    gl.BindVertexArray(VAO);

    var VBO: u32 = undefined;
    //create the vbo
    gl.GenBuffers(1, @ptrCast(&VBO));
    //make this the current active vbo
    gl.BindBuffer(gl.ARRAY_BUFFER, VBO);
    //push the data into it
    gl.BufferData(gl.ARRAY_BUFFER, @sizeOf(f32) * vertices.len, &vertices, gl.STATIC_DRAW);
    //specify how the data inside the vbo is read
    gl.VertexAttribPointer(0, 3, gl.FLOAT, gl.FALSE, 3 * @sizeOf(f32), 0);
    //dont really know
    gl.EnableVertexAttribArray(0);

    var EBO: u32 = undefined;
    gl.GenBuffers(1, @ptrCast(&EBO));
    gl.BindBuffer(gl.ELEMENT_ARRAY_BUFFER, EBO);
    gl.BufferData(gl.ELEMENT_ARRAY_BUFFER, @sizeOf(u32) * indices.len, &indices, gl.STATIC_DRAW);

    return EBO;
}
pub fn handleMovement(appState: *state.app_state) bool {
    // --- WASD & ARROW MOVEMENT LOGIC ---
    const speed: f32 = 0.05; // Adjust this to change movement speed
    var moved_this_frame = false;

    // Forward (W or Up Arrow)
    if (c.glfwGetKey(appState.window, c.GLFW_KEY_W) == c.GLFW_PRESS or
        c.glfwGetKey(appState.window, c.GLFW_KEY_UP) == c.GLFW_PRESS)
    {
        appState.scenes.current_state.cam_pos[0] -= @sin(appState.scenes.current_state.yaw) * speed;
        appState.scenes.current_state.cam_pos[2] -= @cos(appState.scenes.current_state.yaw) * speed;
        moved_this_frame = true;
    }

    // Backward (S or Down Arrow)
    if (c.glfwGetKey(appState.window, c.GLFW_KEY_S) == c.GLFW_PRESS or
        c.glfwGetKey(appState.window, c.GLFW_KEY_DOWN) == c.GLFW_PRESS)
    {
        appState.scenes.current_state.cam_pos[0] += @sin(appState.scenes.current_state.yaw) * speed;
        appState.scenes.current_state.cam_pos[2] += @cos(appState.scenes.current_state.yaw) * speed;
        moved_this_frame = true;
    }

    // Left (A or Left Arrow)
    if (c.glfwGetKey(appState.window, c.GLFW_KEY_A) == c.GLFW_PRESS or
        c.glfwGetKey(appState.window, c.GLFW_KEY_LEFT) == c.GLFW_PRESS)
    {
        appState.scenes.current_state.cam_pos[0] -= @cos(appState.scenes.current_state.yaw) * speed;
        appState.scenes.current_state.cam_pos[2] += @sin(appState.scenes.current_state.yaw) * speed;
        moved_this_frame = true;
    }

    // Right (D or Right Arrow)
    if (c.glfwGetKey(appState.window, c.GLFW_KEY_D) == c.GLFW_PRESS or
        c.glfwGetKey(appState.window, c.GLFW_KEY_RIGHT) == c.GLFW_PRESS)
    {
        appState.scenes.current_state.cam_pos[0] += @cos(appState.scenes.current_state.yaw) * speed;
        appState.scenes.current_state.cam_pos[2] -= @sin(appState.scenes.current_state.yaw) * speed;
        moved_this_frame = true;
    }

    // Up (Space) / Down (Left Shift)
    if (c.glfwGetKey(appState.window, c.GLFW_KEY_SPACE) == c.GLFW_PRESS) {
        appState.scenes.current_state.cam_pos[1] += speed;
        moved_this_frame = true;
    }
    if (c.glfwGetKey(appState.window, c.GLFW_KEY_LEFT_SHIFT) == c.GLFW_PRESS) {
        appState.scenes.current_state.cam_pos[1] -= speed;
        moved_this_frame = true;
    }

    return moved_this_frame;
}
// pub fn handleMovement(appState: *state.app_state) bool {
//     // --- WASD MOVEMENT LOGIC ---
//     const speed: f32 = 0.05; // Adjust this to change movement speed
//     var moved_this_frame = false;
//
//     // Forward (W) / Backward (S)
//     if (c.glfwGetKey(appState.window, c.GLFW_KEY_W) == c.GLFW_PRESS) {
//         appState.scenes.current_state.cam_pos[0] -= @sin(appState.scenes.current_state.yaw) * speed;
//         appState.scenes.current_state.cam_pos[2] -= @cos(appState.scenes.current_state.yaw) * speed;
//         moved_this_frame = true;
//     }
//     if (c.glfwGetKey(appState.window, c.GLFW_KEY_S) == c.GLFW_PRESS) {
//         appState.scenes.current_state.cam_pos[0] += @sin(appState.scenes.current_state.yaw) * speed;
//         appState.scenes.current_state.cam_pos[2] += @cos(appState.scenes.current_state.yaw) * speed;
//         moved_this_frame = true;
//     }
//     // Left (A) / Right (D)
//     if (c.glfwGetKey(appState.window, c.GLFW_KEY_A) == c.GLFW_PRESS) {
//         appState.scenes.current_state.cam_pos[0] -= @cos(appState.scenes.current_state.yaw) * speed;
//         appState.scenes.current_state.cam_pos[2] += @sin(appState.scenes.current_state.yaw) * speed;
//         moved_this_frame = true;
//     }
//     if (c.glfwGetKey(appState.window, c.GLFW_KEY_D) == c.GLFW_PRESS) {
//         appState.scenes.current_state.cam_pos[0] += @cos(appState.scenes.current_state.yaw) * speed;
//         appState.scenes.current_state.cam_pos[2] -= @sin(appState.scenes.current_state.yaw) * speed;
//         moved_this_frame = true;
//     }
//     // Up (Space) / Down (Left Shift)
//     if (c.glfwGetKey(appState.window, c.GLFW_KEY_SPACE) == c.GLFW_PRESS) {
//         appState.scenes.current_state.cam_pos[1] += speed;
//         moved_this_frame = true;
//     }
//     if (c.glfwGetKey(appState.window, c.GLFW_KEY_LEFT_SHIFT) == c.GLFW_PRESS) {
//         appState.scenes.current_state.cam_pos[1] -= speed;
//         moved_this_frame = true;
//     }
//
//     return moved_this_frame;
// }
