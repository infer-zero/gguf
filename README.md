# gguf

Parser for the [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) model format (v3), used by llama.cpp and the broader GGML ecosystem.

## Features

- Complete GGUF v3 format support with magic number and version validation
- Metadata key-value extraction with type coercion (`getUint`, `getFloat`, `getString`)
- Tensor information parsing with quantization type mapping (Q4_0, Q4_1, Q8_0, Q6_K, BF16, FP16, FP32)
- Array metadata types

## Usage

```bash
zig fetch --save git+https://github.com/infer-zero/gguf
```

Then in your `build.zig`:

```zig
const gguf_dep = b.dependency("gguf", .{ .target = target, .optimize = optimize });
my_mod.addImport("gguf", gguf_dep.module("gguf"));
```

```zig
const gguf = @import("gguf");

var model = try gguf.GgufFile.init(allocator, "/path/to/model.gguf");
defer model.deinit();

const arch = model.metadata.getString("general.architecture");
const tensors = model.tensors;
```

## Dependencies

- [base](https://github.com/infer-zero/base) — Core inference abstractions

## License

MIT
