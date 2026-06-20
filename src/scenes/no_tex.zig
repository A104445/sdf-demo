const scene = @import("../scene.zig");
const state = @import("../state.zig");
const gl = @import("gl");
const utils = @import("../utils.zig");
const shaders = @import("../shaders.zig");
const std = @import("std");

pub fn init_metadata() scene.SceneMetadata {
    return scene.SceneMetadata{ .name = "check1-ct" };
}

pub fn init_cam() state.scene_state {
    return .{
        .bound_texture_count = 0,
        .cam_pos = .{ -54.37, -0.10, 39.99 },
        .yaw = -1.01,
        .pitch = 0.15,
    };
}

// Opt-in animation: orbit the camera around the origin exactly once over the duration.
// `anim_duration` controls SPEED (seconds per orbit) independently of FPS (smoothness):
// the phase is normalized so one full revolution always spans [0, anim_duration].
pub const anim_duration: f32 = 12.0; // seconds for one orbit (raise = slower)
pub fn animate(s: *state.scene_state, t: f32) void {
    const tau: f32 = 6.2831853;
    const angle = (t / anim_duration) * tau; // one full orbit over the duration
    const r: f32 = 67.0; // ~ initial distance from origin
    s.cam_pos = .{ @sin(angle) * r, s.cam_pos[1], @cos(angle) * r };
    s.yaw = angle; // turn to keep facing roughly toward the origin as it orbits
}

pub fn init(allocator: std.mem.Allocator) !scene.Scene {
    const vert_path = "./shaders/shader.vert";
    const vert_source = try shaders.readFileToString(allocator, vert_path);
    defer allocator.free(vert_source);
    const compiled_vert = try shaders.compileShader(allocator, vert_source, gl.VERTEX_SHADER);

    const frag_path = "./shaders/ct-newScene1.frag";
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
