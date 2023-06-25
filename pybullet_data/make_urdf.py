#%% 

def make_urdf(file_name, widths = [.6]*10):
    
    text = """<?xml version="1.0"?>

<robot name="{}">

    <!-- The base --> 
    <link name="base_link">
        <inertial>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <mass value="1"/>
            <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <cylinder length="0.01" radius="0.6"/>
            </geometry>
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <cylinder length="0.01" radius="0.6"/>
            </geometry>
        </collision>
    </link>""".format(file_name)
    
    for i, width in enumerate(widths):
        prev = i-1 
        if(prev == -1): prev = "base"
        text += """\n
    <!-- {} -->
    <link name="{}_link">
        <inertial>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <mass value="1"/>
            <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <cylinder length="0.05" radius="{}"/>
            </geometry>
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <cylinder length="0.05" radius="{}"/>
            </geometry>
        </collision>
    </link>

    <!-- Joint from {} to {} -->
    <joint name="{}_{}_joint" type="fixed">
        <parent link="{}_link"/>
        <child link="{}_link"/>
        <origin xyz="0 0 0.05" rpy="0 0 0"/>
    </joint>""".format(i, i, width, width, prev, i, prev, i, prev, i)
    
    text += "\n\n</robot>"
    
    file = open("{}.urdf".format(file_name), "w")
    file.write(text)
    file.close()

make_urdf("1_cylinder",      [.6] * 20)
make_urdf("2_pole",          [.2] * 20)
make_urdf("3_flat_bottom",   [.6] * 1 + [.2] * 19)
make_urdf("4_flat_top",      [.2] * 19 + [.6] * 1)
make_urdf("5_cone",          [.6 - .03*i for i in range(20)])
make_urdf("6_cone_inverted", [.03 + .03*i for i in range(20)])
# %%
