const c = @import("c.zig").c;
const Scene = @import("scene.zig");
const std = @import("std");

pub const mouse_state = struct {
    first_mouse: bool = true,
    is_captured: bool = false,
    imgui_wants_mouse: bool = false,
    moved: bool = false,
    lmb_pressed: bool = false,
    //0 -> x; 1 -> y
    last_pos: [2]f64 = .{ 0, 0 },
};

pub const imgui_state = struct {
    window_size: c.ImVec2 = .{ .x = 0, .y = 0 },
    window_pos: c.ImVec2 = .{ .x = 0, .y = 0 },
    is_active: bool = false,
};

pub const scene_state = struct {
    cam_pos: [3]f32 = .{ 0, 0, 0 },
    yaw: f32 = 0,
    pitch: f32 = 0,
    bound_texture_count: u32 = 0,
    aperture: f32 = 0,
    focal_distance: f32 = 0.1,
};

pub const SceneManager = struct {
    allocator: std.mem.Allocator,
    registry: []const Scene.SceneEntry,
    active_scene: ?Scene.Scene = null,
    active_index: usize = 0,
    requested_scene_index: c_int = 0,
    current_state: scene_state = .{},

    pub fn switchScene(self: *SceneManager, requested_index: usize) !void {
        if (self.active_scene) |*scene| {
            try scene.deinit(self.allocator);
        }

        self.active_scene = try self.registry[requested_index].init_fn(self.allocator);
        self.active_index = requested_index;
        self.current_state = self.registry[requested_index].init_cam_fn();

        self.requested_scene_index = @intCast(requested_index);
    }
};

pub const metrics_state = struct {
    ms_per_frame: f64 = 0,
    last_time: f64 = 0,
};

pub const renderer_state = struct {
    total_accumulated_frames: u32 = 0,
    samples_per_frame: c_int = 0,
    should_reset_accumulation: bool = false,
    want_to_save: bool = false,
    render_w: c_int = 0,
    render_h: c_int = 0,

    // Animation playback (foundation): a controlled clock the renderer holds constant
    // for one output frame's accumulation, then steps. See main loop / animate hook.
    anim_playing: bool = false, // playback toggle
    anim_time: f32 = 0, // the controlled clock (seconds), frozen per frame
    anim_fps: f32 = 24, // render setting: frames per second of the output
    // accumulation iterations per animation frame before advancing the clock.
    // Distinct from samples_per_frame/u_spf (GPU samples within one draw).
    samples_per_anim_frame: u32 = 128,

    // Offline render-to-video (pipes converged frames to ffmpeg -> render/<scene>.mp4).
    anim_rendering: bool = false, // a render is in progress
    anim_frame_index: u32 = 0, // frame currently being rendered
    anim_total_frames: u32 = 0, // = round(anim_duration * anim_fps)
    render_crf: c_int = 17, // h264 quality (lower = better; 0 ~ lossless)
    want_to_render: bool = false, // UI -> loop: start a render (mirrors want_to_save)
    want_cancel_render: bool = false, // UI -> loop: cancel the in-progress render
};

pub const app_state = struct {
    mouse: mouse_state = .{},
    imgui: imgui_state = .{},
    scenes: SceneManager,
    metrics: metrics_state = .{},
    renderer: renderer_state = .{},
    window_w: c_int = 0,
    window_h: c_int = 0,
    window_title: []const u8 = "",
    window: ?*c.struct_GLFWwindow = @ptrFromInt(0),
};
