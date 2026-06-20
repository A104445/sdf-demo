const std = @import("std");

// Wraps an ffmpeg subprocess that reads raw RGBA frames from stdin and encodes them
// into render/<scene>.mp4. ffmpeg is a hard dependency — start() returns an error
// (and the caller may crash) if it isn't on PATH.
pub const Encoder = struct {
    child: std.process.Child,

    pub fn start(
        allocator: std.mem.Allocator,
        w: c_int,
        h: c_int,
        fps: f32,
        crf: c_int,
        scene_name: []const u8,
    ) !Encoder {
        // Ensure the output directory exists (matches the existing render/ convention).
        try std.fs.cwd().makePath("render");

        // The dynamic argv strings only need to live across spawn(); an arena keeps it tidy.
        var arena_state = std.heap.ArenaAllocator.init(allocator);
        defer arena_state.deinit();
        const a = arena_state.allocator();

        const size = try std.fmt.allocPrint(a, "{d}x{d}", .{ w, h });
        const fps_str = try std.fmt.allocPrint(a, "{d}", .{fps});
        const crf_str = try std.fmt.allocPrint(a, "{d}", .{crf});
        const out_path = try std.fmt.allocPrint(a, "render/{s}.mp4", .{scene_name});

        const argv = [_][]const u8{
            "ffmpeg",        "-y",
            // describe the raw frames we feed over stdin
            "-f",            "rawvideo",
            "-pixel_format", "rgba",
            "-video_size",   size,
            "-framerate",    fps_str,
            "-i",            "-",
            // flip (GL is bottom-up) + force even dims for yuv420p
            "-vf",           "vflip,scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-pix_fmt",      "yuv420p",
            "-crf",          crf_str,
            "-an",           out_path,
        };

        var child = std.process.Child.init(&argv, allocator);
        child.stdin_behavior = .Pipe;
        child.stdout_behavior = .Inherit;
        child.stderr_behavior = .Inherit;
        try child.spawn();

        return Encoder{ .child = child };
    }

    pub fn writeFrame(self: *Encoder, pixels: []const u8) !void {
        try self.child.stdin.?.writeAll(pixels);
    }

    // Closes stdin (signals EOF so ffmpeg finalizes the file) and waits for it to exit.
    pub fn finish(self: *Encoder) !void {
        if (self.child.stdin) |stdin| {
            stdin.close();
            self.child.stdin = null; // prevent wait() from double-closing
        }
        _ = try self.child.wait();
    }
};
