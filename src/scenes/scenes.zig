const Scene = @import("../scene.zig");

pub const all = struct {
    pub const no_tex = @import("no_tex.zig");
    //pub const sanity = @import("sanity.zig");
    //pub const scene1 = @import("scene1.zig");
    pub const scene1_test = @import("scene1-test.zig");
    //pub const no_tex_before = @import("scene2.zig");
    //pub const testing_scene = @import("testing_scene.zig");
    pub const melt = @import("melt.zig");
};

pub const registry: []const Scene.SceneEntry = blk: {
    const decls = @typeInfo(all).@"struct".decls;
    var list: []const Scene.SceneEntry = &.{};
    for (decls) |d| {
        const mod = @field(all, d.name);
        list = list ++ &[_]Scene.SceneEntry{.{
            .metadata = mod.init_metadata(),
            .init_fn = mod.init,
            .init_cam_fn = mod.init_cam,
            // Optional animation members — a scene opts in just by declaring them.
            .animate_fn = if (@hasDecl(mod, "animate")) mod.animate else null,
            .anim_duration = if (@hasDecl(mod, "anim_duration")) mod.anim_duration else 0,
        }};
    }
    break :blk list;
};
