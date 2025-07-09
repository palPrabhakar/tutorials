# SPIR-V Specification Presentation

## Introduction to SPIR-V

* SPIR-V is a binary intermediate representation (IR) used for graphical and compute shaders.
* Standardized by the Khronos Group.
* Targeted by high-level languages like GLSL, HLSL, and OpenCL C.
* Enables portability across APIs such as Vulkan, OpenCL, and WebGPU.

**Example:** GLSL to SPIR-V: `vec4 color = texture(uSampler, uv);` gets compiled to a sequence of SPIR-V instructions with `OpImageSampleImplicitLod`.

---

## Why SPIR-V?

* Cross-language and cross-platform IR.
* Compact binary format allows fast loading and smaller footprint.
* Enables easy validation, optimization, and instrumentation.
* Facilitates unified shader compilation workflows.

**Example:** Instead of shipping GLSL, developers ship compiled SPIR-V for faster load times and better obfuscation.

---

## SPIR-V Module Structure

* **Header** includes:

  * Magic number (0x07230203)
  * SPIR-V version (e.g., 0x00010300 for 1.3)
  * Generator tool ID (e.g., 0 for unknown)
  * Bound (maximum ID used + 1)
  * Reserved word (always 0)

* **Logical layout** (must follow spec-defined order):

  1. OpCapability
  2. OpExtension (optional)
  3. OpExtInstImport (optional)
  4. OpMemoryModel
  5. OpEntryPoint
  6. OpExecutionMode (optional)
  7. Debug and Annotation Instructions (e.g., OpName, OpDecorate)
  8. Types, constants, and global variables (OpType\*, OpConstant, OpVariable)
  9. Functions (OpFunction ... OpFunctionEnd)

**Concrete Example:** A minimal fragment shader module (disassembled):

```spirv
; SPIR-V
; Version: 1.0
; Generator: Khronos Glslang Reference Front End; 10
; Bound: 11
; Schema: 0
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %4 "main" %9
OpExecutionMode %4 OriginUpperLeft
OpName %4 "main"
OpName %9 "fragColor"
OpDecorate %9 Location 0

%2 = OpTypeVoid
%3 = OpTypeFunction %2
%6 = OpTypeFloat 32
%7 = OpTypeVector %6 4
%8 = OpTypePointer Output %7
%9 = OpVariable %8 Output
%10 = OpConstantComposite %7 %6_1 %6_1 %6_1 %6_1

%4 = OpFunction %2 None %3
%5 = OpLabel
OpStore %9 %10
OpReturn
OpFunctionEnd
```

This example writes `vec4(1.0)` to a fragment output. It declares types, capabilities, entry points, and a function body with `OpStore`.
````
OpEntryPoint Fragment %main "main" %fragColor
````

---

## Strong Static Typing & SSA

- SPIR-V enforces strong static typing, meaning all values and operations must have explicitly declared types.
- There are no implicit conversions — types must match exactly for operations.
- All values are SSA (Static Single Assignment) IDs:
  - Each ID is assigned exactly once.
  - Values flow explicitly through the control flow graph.
- Phi nodes (`OpPhi`) reconcile values coming from different predecessors.
- SSA enables powerful compiler optimizations like constant folding and dead code elimination.

**Benefits:**
- Ensures predictable semantics across different compilers and drivers.
- Simplifies validation and transformation passes.
- Guarantees explicit and analyzable data flow.

**Example:** Declaration and use of integer, float, and pointer types:

```spirv
%int = OpTypeInt 32 1
%float = OpTypeFloat 32
%ptr_output = OpTypePointer Output %vec4
```

---

## Instruction Encoding

* All instructions are 32-bit word sequences.
* First word: `[word count << 16] | opcode`
* Subsequent words: operands (IDs, literals, strings)

**Example:**

```
0x00020011  // word count = 2, opcode = 17 (OpMemoryModel)
0x00000001  // Logical memory model
0x00000001  // GLSL450 execution model
```

---

## Control Flow

* Structured control flow enforced via:

  * OpSelectionMerge (for if-else)
  * OpLoopMerge (for loops)
* Uses basic blocks with labels and SSA phi nodes.
* Unstructured control flow is not valid in SPIR-V shaders.

**Example:**

```spirv
OpSelectionMerge %merge None
OpBranchConditional %cond %true_label %false_label
```

---

## Memory Model

* Must declare memory and execution models (e.g., Logical, GLSL450).
* Storage classes:

  * Input, Output, Uniform, PushConstant, Workgroup, Private, etc.
* Supports memory barriers and synchronization for compute shaders.

**Example:**

```spirv
OpMemoryBarrier Workgroup UniformMemory
OpControlBarrier Workgroup Workgroup AcquireRelease
```

---

## Decorations & Metadata

* Add metadata via `OpDecorate`, `OpMemberDecorate`.
* Common use cases:

  * Specify layout: `Location`, `Binding`, `Offset`
  * Optimization hints: `RelaxedPrecision`, `Invariant`

**Example:**

```spirv
OpDecorate %fragColor Location 0
OpDecorate %ubo DescriptorSet 0
OpDecorate %ubo Binding 1
```

---

## Capabilities & Extensions

* Capabilities must be declared to use specific features.
* Examples:

  * Shader (basic functionality)
  * Geometry, Tessellation, Int64, Float64
* Extensions provide non-core or vendor-specific features.

**Example:**

```spirv
OpCapability Shader
OpCapability Float64
OpExtension "SPV_KHR_shader_ballot"
```

---

## SPIR-V and GLSL

* GLSLang compiles GLSL → SPIR-V.
* Many constructs map 1:1:

  * `layout(location = 0) out vec4 color;` → `OpDecorate %color Location 0`
* Some features require rewrite (e.g., dynamic indexing, non-square matrices).
* SPIR-V enforces structure GLSL doesn’t always guarantee.

**Example (GLSL to SPIR-V):**

```glsl
layout(location = 0) out vec4 color;
void main() {
    color = vec4(1.0);
}
```

```spirv
OpDecorate %color Location 0
%main = OpFunction ...
OpStore %color %const_vec4_1
```
---

-----

### From GLSL to SPIR-V: Under the Hood

The translation from GLSL to SPIR-V is a sophisticated process. Here's a deeper look at how key concepts are mapped.

### Mapping Capabilities

A SPIR-V module must explicitly declare the features it requires. `glslangValidator` infers these from the GLSL code.

  * **Shader Stage**: Writing a geometry shader (`#pragma shader_stage(geometry)`) adds `OpCapability Geometry`.
  * **Data Types**: Using a `mat4` requires `OpCapability Matrix`. Using 64-bit doubles (`dvec3`) requires `OpCapability Float64`.
  * **Built-in Variables**: Accessing `gl_ClipDistance` requires `OpCapability ClipDistance`.
  * **Texture Operations**: Using texture sampling functions that require explicit derivatives (e.g., `textureGrad`) will add `OpCapability DerivativeControl`.

### Mapping Instructions and Types

GLSL's expression-based syntax is broken down into a sequence of explicit SPIR-V instructions.

  * **Type Declarations**: Every GLSL type gets a corresponding `OpType` declaration in SPIR-V.

      * `vec3` -\> `%v3float = OpTypeVector %float 3`
      * `struct Light { vec3 pos; float intensity; };` -\> `%Light = OpTypeStruct %v3float %float`

  * **Variable Declarations**: GLSL variables become `OpVariable` instructions, which define their type and, crucially, their **Storage Class**.

      * `in vec3 normal;` -\> `OpVariable %ptr_v3_input Input`
      * `uniform sampler2D tex;` -\> `OpVariable %ptr_sampler UniformConstant`
      * `vec3 temp = ...;` -\> `OpVariable %ptr_v3_function Function`

  * **Operations**: GLSL expressions are unrolled into individual instructions that operate on result IDs.

    **GLSL**: `vec4 color = texture(tex, uv) * baseColor;`

    **SPIR-V (Conceptual)**:

    ```
    %sampler = OpLoad %type_sampler %ptr_tex
    %coords = OpLoad %v2float %ptr_uv
    %sampled_image = OpSampledImage %type_sampled_image %sampler %image
    %texel = OpImageSampleImplicitLod %v4float %sampled_image %coords
    %base = OpLoad %v4float %ptr_baseColor
    %result = OpFMul %v4float %texel %base
    OpStore %ptr_color %result
    ```

### Handling Variables and the SSA Form

**SPIR-V** is in **Static Single Assignment (SSA)** form, meaning every result ID is assigned a value exactly once. GLSL allows variables to be reassigned freely. The compiler bridges this gap using `OpPhi` instructions.

Consider this GLSL snippet:

```glsl
vec3 color = vec3(1.0, 0.0, 0.0); // red
if (condition) {
    color = vec3(0.0, 1.0, 0.0); // green
}
// ... use color
```

The variable `color` is assigned in two different places. In SPIR-V, this is handled by creating different control flow blocks and merging the results.

  * **Block 1 (Before if)**: A result ID `%red` is created for `vec3(1.0, 0.0, 0.0)`.

  * **Block 2 (Inside if)**: A result ID `%green` is created for `vec3(0.0, 1.0, 0.0)`.

  * **Block 3 (After if)**: An `OpPhi` instruction selects the final value.

    ```
    ; Block 3 (Merge Block)
    %final_color = OpPhi %v3float   %red %Block1   %green %Block2
    ```

The `OpPhi` instruction says: `%final_color` will be `%red` if control came from Block1, or `%green` if control came from Block2. This elegantly transforms the mutable GLSL variable into a single, immutable SPIR-V result ID.

---

## Comparison with Other IRs

* **SPIR-V vs LLVM IR:**

  * SPIR-V is explicitly structured, binary, GPU-oriented.
  * LLVM IR is textual, CPU-focused, flexible and dynamic.
* **SPIR-V vs DXIL:**

  * DXIL is LLVM-based for DirectX shaders.
  * SPIR-V is designed independently with Khronos governance.

---

## Ecosystem & Tools

* **SPIRV-Tools**:

  * `spirv-as`, `spirv-dis`, `spirv-val`, `spirv-opt`
* **SPIRV-Cross**:

  * Converts SPIR-V to GLSL, HLSL, MSL
* **RenderDoc**, **GDB**, Vulkan SDK validation layers
* DebugInfo support via extensions (e.g., `NonSemantic.Shader.DebugInfo.100`)

---

## Use in Compiler Development

* Frontends like GLSLang or Clspv emit SPIR-V.
* Backends may consume SPIR-V for execution or further lowering.
* IR optimization: dead code elimination, constant folding via `spirv-opt`
* Supports specialization constants for configurable pipelines.

---
