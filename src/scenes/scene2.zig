const scene = @import("../scene.zig");
const state = @import("../state.zig");
const gl = @import("gl");
const utils = @import("../utils.zig");
const shaders = @import("../shaders.zig");
const std = @import("std");

pub fn init_metadata() scene.SceneMetadata {
    return scene.SceneMetadata{ .name = "no_text-before" };
}

pub fn init_cam() state.scene_state {
    return .{
        .bound_texture_count = 0,
        .cam_pos = .{ 0.0, 0.0, 0.0 },
        .yaw = 0.0,
        .pitch = 0.0,
    };
}

pub fn init(allocator: std.mem.Allocator) !scene.Scene {
    const vert_path = "./shaders/shader.vert";
    const vert_source = try shaders.readFileToString(allocator, vert_path);
    defer allocator.free(vert_source);
    const compiled_vert = try shaders.compileShader(allocator, vert_source, gl.VERTEX_SHADER);

    const frag_path = "./shaders/ct-newScene2.frag";
    const frag_source = try shaders.readFileToString(allocator, frag_path);
    defer allocator.free(frag_source);
    const compiled_frag = try shaders.compileShader(allocator, frag_source, gl.FRAGMENT_SHADER);

    const shader_program = try shaders.setupShaderProgram(allocator, &.{ compiled_vert, compiled_frag });

    const texture_paths = &[_][:0]const u8{};

    const texture_names = &[_][:0]const u8{};

    const textures = try allocator.alloc(u32, texture_paths.len);
    errdefer allocator.free(textures);

    gl.GenTextures(texture_paths.len, textures.ptr);

    const self: scene.Scene = .{
        .shader_program = shader_program,
        .texture_names = texture_names,
        .textures = textures,
    };

    for (texture_paths, 0..) |path, i| {
        try utils.loadTexture(path, textures[i]);
    }

    return self;
}
