#%%

class Part:
    def __init__(
        self, 
        name, 
        mass = 0, 
        shape = "box",
        size = (1, 1, 1), 
        joint_parent = None, 
        joint_origin = (0, 0, 0), 
        joint_axis = (1, 0, 0),
        joint_type = "fixed",
        sensors = 0, 
        sensor_width = .02, 
        sensor_angle = 0,
        sensor_sides = ["start", "stop", "top", "bottom", "left", "right"],
        joint_rpy=(0, 0, 0),
        joint_limits = [0, 0, 0, 0]):
        
        params = locals()
        for param in params:
            if param != "self":
                setattr(self, param, params[param])
                
        self.sensor_positions = []
        self.sensor_dimensions = []
        self.sensor_angles = []
        
        self.shape_text = self.get_shape_text()
        self.sensor_text = ""  # Initialize as empty
        self.joint_text = self.get_joint_text()
        
    def get_text(self):
        return(self.shape_text + self.sensor_text + self.joint_text)
        
    def get_shape_text(self):
        if(self.shape == "box"):
            shape_sizes = f'box size="{self.size[0]} {self.size[1]} {self.size[2]}"'
        if(self.shape == "cylinder"):
            shape_sizes = f'cylinder radius="{self.size[0]}" length="{self.size[1]}"'          
        return(
f"""\n\n
    <!-- {self.name} -->
    <link name="{self.name}">
        <inertial>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <mass value="{self.mass}"/>
            <inertia ixx=".1" ixy=".1" ixz=".1" iyy=".1" iyz=".1" izz=".1"/>
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <{shape_sizes}/>
            </geometry>
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <{shape_sizes}/>
            </geometry>
        </collision>
    </link>""")
        
    def make_sensor(self, i, side, size, origin, minus, first_plus, second_plus, parts):
        sensor_origin = [
            o if (i + first_plus) % 3 == self.sensor_angle 
            else o - self.size[i]/2 if (i + second_plus) % 3 == self.sensor_angle and minus 
            else o + self.size[i]/2 if (i + second_plus) % 3 == self.sensor_angle and not minus 
            else o
            for i, o in enumerate(origin)]
        
        # Apply cumulative transformations
        parent_part = self
        cumulative_origin = [0, 0, 0]
        while parent_part:
            cumulative_origin = [cumulative_origin[j] + parent_part.joint_origin[j] for j in range(3)]
            parent_part = next((part for part in parts if part.name == parent_part.joint_parent), None)
        
        transformed_position = [cumulative_origin[j] + sensor_origin[j] for j in range(3)]
        
        self.sensor_positions.append(transformed_position)
        self.sensor_dimensions.append(size)
        self.sensor_angles.append(self.joint_rpy)
                
        sensor = Part(
            name = f"{self.name}_sensor_{i}_{side}",
            mass = 0,
            size = size,
            joint_parent = f"{self.name}",
            joint_origin = sensor_origin,
            joint_axis = self.joint_axis,
            joint_type = "fixed")
        
        return(sensor.get_text())
    
    def get_sensors_text(self, parts):
        if(self.sensors == 0):
            return("")
        text = ""
        if(self.sensors == 1):
            origin = (0, 0, 0)
        else:
            origin = (
                -self.size[0]/2 + self.size[0]/(2*self.sensors) if self.sensor_angle == 0 else 0, 
                -self.size[1]/2 + self.size[1]/(2*self.sensors) if self.sensor_angle == 1 else 0, 
                -self.size[2]/2 + self.size[2]/(2*self.sensors) if self.sensor_angle == 2 else 0)
            
        start_stop_size = [
            s if (i + 2) % 3 == self.sensor_angle 
            else s if (i + 1) % 3 == self.sensor_angle
            else self.sensor_width 
            for i, s in enumerate(self.size)]
        
        top_bottom_size = [
            s / self.sensors if (i + 0) % 3 == self.sensor_angle 
            else s if (i + 2) % 3 == self.sensor_angle
            else self.sensor_width 
            for i, s in enumerate(self.size)]
        
        left_right_size = [
            s / self.sensors if (i + 0) % 3 == self.sensor_angle 
            else s if (i + 1) % 3 == self.sensor_angle
            else self.sensor_width 
            for i, s in enumerate(self.size)]
        
        for i in range(self.sensors):
            if(i == 0 and "start" in self.sensor_sides):
                text += self.make_sensor(i, "start", start_stop_size, (0, 0, 0), False, 2, 0, parts)
            if(i == self.sensors - 1 and "stop" in self.sensor_sides):
                text += self.make_sensor(i, "stop", start_stop_size, (0, 0, 0), True, 2, 0, parts)
            if("top" in self.sensor_sides):
                text += self.make_sensor(i, "top", top_bottom_size, origin, False, 2, 1, parts)
            if("bottom" in self.sensor_sides):
                text += self.make_sensor(i, "bottom", top_bottom_size, origin, True, 2, 1, parts)
            if("left" in self.sensor_sides):
                text += self.make_sensor(i, "left", left_right_size, origin, False, 0, 2, parts)
            if("right" in self.sensor_sides):
                text += self.make_sensor(i, "right", left_right_size, origin, True, 0, 2, parts)
            origin = (
                    origin[0] + self.size[0]/(self.sensors) if self.sensor_angle == 0 else origin[0], 
                    origin[1] + self.size[1]/(self.sensors) if self.sensor_angle == 1 else origin[1],
                    origin[2] + self.size[2]/(self.sensors) if self.sensor_angle == 2 else origin[2])
        return(text)
        
    def get_joint_text(self):
        if self.joint_parent is None:
            return ""
        # Use self.joint_rpy here instead of fixed 0 0 0
        return f"""
    <!-- Joint: {self.joint_parent}, {self.name} -->
    <joint name="{self.joint_parent}_{self.name}_joint" type="{self.joint_type}">
        <parent link="{self.joint_parent}"/>
        <child link="{self.name}"/>
        <origin xyz="{self.joint_origin[0]} {self.joint_origin[1]} {self.joint_origin[2]}"
                rpy="{self.joint_rpy[0]} {self.joint_rpy[1]} {self.joint_rpy[2]}"/>
        <axis xyz="{self.joint_axis[0]} {self.joint_axis[1]} {self.joint_axis[2]}"/>
        {"" if self.joint_type == "fixed" else f'<limit lower="{self.joint_limits[0]}" upper="{self.joint_limits[1]}" effort="{self.joint_limits[2]}" velocity="{self.joint_limits[3]}"/>'}
    </joint>"""
    
    
    
if(__name__ == "__main__"):
    part = Part(
        name = "body", 
        mass = 100, 
        size = (1, 1, 1),
        sensors = 1,
        sensor_sides = ["start", "stop", "top", "left", "right"]),