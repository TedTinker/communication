#%%

division_epochs = [100, 150, 450]

def map_epochs_to_xs(division_epochs, xs):
    length_xs = len(xs)
    mapped_values = []
    for epoch in division_epochs:
        position = (epoch / division_epochs[-1]) * (length_xs - 1)
        index = round(position)
        mapped_values.append(xs[index])
    return mapped_values


    
xs_1 = [x for x in range(450)]
print(len(xs_1))
xs_2 = [x for x in range(45)]
print(len(xs_2))
    
map_epochs_to_xs(division_epochs, xs_1)
map_epochs_to_xs(division_epochs, xs_2)