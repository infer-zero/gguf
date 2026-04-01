const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const base_dep = b.dependency("infer_base", .{ .target = target, .optimize = optimize });
    const base_mod = base_dep.module("infer_base");

    const mod = b.addModule("gguf", .{
        .root_source_file = b.path("src/gguf.zig"),
        .target = target,
        .optimize = optimize,
    });
    mod.addImport("base", base_mod);

    const tests = b.addTest(.{
        .root_module = mod,
    });

    const run_tests = b.addRunArtifact(tests);
    run_tests.setCwd(b.path("."));
    const test_step = b.step("test", "Run gguf tests");
    test_step.dependOn(&run_tests.step);
}
