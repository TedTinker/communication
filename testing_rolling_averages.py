#%%

import matplotlib.pyplot as plt

# Define the rolling average function
def rolling_average(lst, window_size=500):
    new_list = [0 if lst[0] is None else float(lst[0])]
    for i in range(1, len(lst)):
        if lst[i] is None:
            new_list.append(new_list[-1])
        else:
            start_index = max(0, i - window_size + 1)
            window = [x for x in lst[start_index:i+1] if x is not None]
            if window:
                new_value = sum(window) / len(window)
            else:
                new_value = 0 
            new_list.append(new_value)
    return new_list

# Define test cases
test_cases = [
    ([None] + [10] * 10 + [None] + [10] * 10, 3),                    # Small list with small window
]

# Plot each test case
for i, (data, window) in enumerate(test_cases, start=1):
    rolling_avg = rolling_average(data, window_size=window)
    plt.figure(figsize=(10, 4))
    plt.plot(data, label="Original Data", marker='o', linestyle="--")
    plt.plot(rolling_avg, label=f"Rolling Average (window={window})")
    plt.title(f"Test Case {i}")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

#%%

import torch

def ignore_silence(values, voice, pred_voice):
    # Get the max indices along the first dimension (dim=2) for both voice and pred_voice
    max_indices_voice = torch.argmax(voice, dim=2)
    max_indices_pred_voice = torch.argmax(pred_voice, dim=2)
    
    # Create silence masks for both voice and pred_voice
    silence_mask_voice = max_indices_voice[:, :, 0] == 0
    silence_mask_pred_voice = max_indices_pred_voice[:, :, 0] == 0
    
    # Combine the masks: we only set values to zero if both masks are True
    combined_silence_mask = silence_mask_voice & silence_mask_pred_voice
    
    # Expand the mask to match the shape of values for broadcasting
    combined_silence_mask_expanded = combined_silence_mask.unsqueeze(-1)
    
    # Apply the combined mask to set elements in values to zero where both masks are True
    values = values * ~combined_silence_mask_expanded
    return values

# Testing the function with sample data
# Define a tensor "values" of shape [2, 2, 1]
values = torch.tensor([[[1.0], [1.0]], [[1.0], [1.0]]])

# Define "voice" tensor of shape [2, 2, 3, 5]
voice = torch.tensor([
    [[[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
     [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]],
    
    [[[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
     [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]]]
])

# Define "pred_voice" tensor of shape [2, 2, 3, 5]
pred_voice = torch.tensor([
    [[[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
     [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]]],
    
    [[[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
     [[0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]]]
])

# Expected output:
# The first entry in values[0][0] should remain 1 because in pred_voice[0][0] the max is not at index 0 in all cases.
# values[0][1] should be 0 because both voice[0][1] and pred_voice[0][1] have max at index 0.
# values[1][0] should be 0 because both voice[1][0] and pred_voice[1][0] have max at index 0.
# values[1][1] should remain 1 because pred_voice[1][1] does not have max at index 0.

# Applying ignore_silence function
result = ignore_silence(values, voice, pred_voice)
print("Modified values after applying ignore_silence:\n", result)


# %%
