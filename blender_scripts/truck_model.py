"""
Blender Script: 3D Truck Model Generator

This script creates a detailed 3D model of a heavy-duty truck with trailer
for simulation and visualization purposes.

Usage:
    blender --background --python truck_model.py

Author: Self-Driving Truck Project
Date: 2025-11-17
"""

import bpy
import math
import mathutils
from typing import Tuple, List


class TruckModelGenerator:
    """Generate 3D truck model with cab and trailer"""

    def __init__(self):
        self.clear_scene()
        self.truck_length = 7.0  # meters
        self.truck_width = 2.5
        self.truck_height = 3.5
        self.trailer_length = 13.6  # Standard 45ft trailer
        self.trailer_width = 2.5
        self.trailer_height = 4.0

    def clear_scene(self):
        """Remove all objects from scene"""
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

    def create_material(self, name: str, color: Tuple[float, float, float, float]):
        """Create a material with specified color"""
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes

        # Clear default nodes
        nodes.clear()

        # Add shader nodes
        output = nodes.new('ShaderNodeOutputMaterial')
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')

        # Set color
        bsdf.inputs['Base Color'].default_value = color
        bsdf.inputs['Metallic'].default_value = 0.3
        bsdf.inputs['Roughness'].default_value = 0.4

        # Connect nodes
        mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

        return mat

    def create_truck_cab(self) -> bpy.types.Object:
        """Create truck cab (tractor unit)"""
        # Main cab body
        bpy.ops.mesh.primitive_cube_add(
            size=1,
            location=(0, 0, self.truck_height/2)
        )
        cab = bpy.context.active_object
        cab.name = "Truck_Cab"
        cab.scale = (self.truck_length, self.truck_width, self.truck_height)

        # Apply scale
        bpy.ops.object.transform_apply(scale=True)

        # Add material
        mat = self.create_material("Cab_Material", (0.8, 0.1, 0.1, 1.0))  # Red
        cab.data.materials.append(mat)

        return cab

    def create_windshield(self, cab: bpy.types.Object):
        """Create windshield on truck cab"""
        bpy.ops.mesh.primitive_cube_add(
            size=1,
            location=(self.truck_length/2 - 0.2, 0, self.truck_height - 0.5)
        )
        windshield = bpy.context.active_object
        windshield.name = "Windshield"
        windshield.scale = (0.3, self.truck_width * 0.9, 1.5)

        # Make transparent material
        mat = self.create_material("Glass_Material", (0.5, 0.7, 0.9, 0.3))
        mat.blend_method = 'BLEND'
        windshield.data.materials.append(mat)

        # Parent to cab
        windshield.parent = cab

    def create_wheel(self, location: Tuple[float, float, float], name: str) -> bpy.types.Object:
        """Create a truck wheel"""
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.5,
            depth=0.3,
            location=location,
            rotation=(0, math.pi/2, 0)
        )
        wheel = bpy.context.active_object
        wheel.name = name

        # Add tire material
        mat = self.create_material(f"{name}_Material", (0.1, 0.1, 0.1, 1.0))  # Black
        wheel.data.materials.append(mat)

        return wheel

    def create_truck_wheels(self, cab: bpy.types.Object):
        """Create all wheels for truck cab"""
        wheel_positions = [
            # Front axle
            (self.truck_length/2 - 1.5, self.truck_width/2 + 0.2, 0.5, "Front_Wheel_Right"),
            (self.truck_length/2 - 1.5, -self.truck_width/2 - 0.2, 0.5, "Front_Wheel_Left"),
            # Rear axle
            (-self.truck_length/2 + 1.5, self.truck_width/2 + 0.2, 0.5, "Rear_Wheel_Right_1"),
            (-self.truck_length/2 + 1.5, -self.truck_width/2 - 0.2, 0.5, "Rear_Wheel_Left_1"),
            (-self.truck_length/2 + 0.8, self.truck_width/2 + 0.2, 0.5, "Rear_Wheel_Right_2"),
            (-self.truck_length/2 + 0.8, -self.truck_width/2 - 0.2, 0.5, "Rear_Wheel_Left_2"),
        ]

        for x, y, z, name in wheel_positions:
            wheel = self.create_wheel((x, y, z), name)
            wheel.parent = cab

    def create_trailer(self) -> bpy.types.Object:
        """Create trailer box"""
        trailer_x = -(self.truck_length/2 + self.trailer_length/2 + 2.0)  # Gap between cab and trailer

        bpy.ops.mesh.primitive_cube_add(
            size=1,
            location=(trailer_x, 0, self.trailer_height/2)
        )
        trailer = bpy.context.active_object
        trailer.name = "Trailer"
        trailer.scale = (self.trailer_length, self.trailer_width, self.trailer_height)

        # Apply scale
        bpy.ops.object.transform_apply(scale=True)

        # Add material
        mat = self.create_material("Trailer_Material", (0.9, 0.9, 0.9, 1.0))  # White
        trailer.data.materials.append(mat)

        return trailer

    def create_trailer_wheels(self, trailer: bpy.types.Object):
        """Create trailer wheels"""
        trailer_x = -(self.truck_length/2 + self.trailer_length/2 + 2.0)

        wheel_positions = [
            # Front axle
            (trailer_x + self.trailer_length/2 - 2.0, self.trailer_width/2 + 0.2, 0.5, "Trailer_Wheel_Right_1"),
            (trailer_x + self.trailer_length/2 - 2.0, -self.trailer_width/2 - 0.2, 0.5, "Trailer_Wheel_Left_1"),
            # Middle axle
            (trailer_x + self.trailer_length/2 - 3.5, self.trailer_width/2 + 0.2, 0.5, "Trailer_Wheel_Right_2"),
            (trailer_x + self.trailer_length/2 - 3.5, -self.trailer_width/2 - 0.2, 0.5, "Trailer_Wheel_Left_2"),
            # Rear axle
            (trailer_x + self.trailer_length/2 - 5.0, self.trailer_width/2 + 0.2, 0.5, "Trailer_Wheel_Right_3"),
            (trailer_x + self.trailer_length/2 - 5.0, -self.trailer_width/2 - 0.2, 0.5, "Trailer_Wheel_Left_3"),
        ]

        for x, y, z, name in wheel_positions:
            wheel = self.create_wheel((x, y, z), name)
            wheel.parent = trailer

    def create_fifth_wheel(self, cab: bpy.types.Object, trailer: bpy.types.Object):
        """Create fifth wheel coupling mechanism"""
        coupling_x = -self.truck_length/2 - 1.0

        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.3,
            depth=0.5,
            location=(coupling_x, 0, 1.2),
            rotation=(math.pi/2, 0, 0)
        )
        fifth_wheel = bpy.context.active_object
        fifth_wheel.name = "Fifth_Wheel"

        # Add material
        mat = self.create_material("Metal_Material", (0.5, 0.5, 0.5, 1.0))
        fifth_wheel.data.materials.append(mat)

        fifth_wheel.parent = cab

        # Create constraint for trailer rotation
        constraint = trailer.constraints.new('COPY_ROTATION')
        constraint.target = cab
        constraint.use_x = False
        constraint.use_y = False
        constraint.use_z = True  # Only allow rotation around Z axis

    def create_camera_sensor(self, location: Tuple[float, float, float],
                            rotation: Tuple[float, float, float], name: str):
        """Create camera sensor visualization"""
        bpy.ops.mesh.primitive_cube_add(
            size=0.2,
            location=location
        )
        camera_box = bpy.context.active_object
        camera_box.name = f"Camera_{name}"
        camera_box.rotation_euler = rotation

        # Add camera material (blue)
        mat = self.create_material(f"Camera_{name}_Material", (0.1, 0.3, 0.8, 1.0))
        camera_box.data.materials.append(mat)

        return camera_box

    def create_lidar_sensor(self, location: Tuple[float, float, float], name: str):
        """Create LiDAR sensor visualization"""
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.15,
            depth=0.2,
            location=location
        )
        lidar = bpy.context.active_object
        lidar.name = f"LiDAR_{name}"

        # Add LiDAR material (green)
        mat = self.create_material(f"LiDAR_{name}_Material", (0.1, 0.8, 0.3, 1.0))
        lidar.data.materials.append(mat)

        return lidar

    def add_sensors(self, cab: bpy.types.Object):
        """Add all sensor visualizations to truck"""
        # Front cameras
        front_camera_z = self.truck_height - 0.5

        cameras = [
            ((self.truck_length/2, 0, front_camera_z), (0, 0, 0), "Front_Center"),
            ((self.truck_length/2 - 0.3, self.truck_width/2, front_camera_z), (0, 0, -math.pi/6), "Front_Right"),
            ((self.truck_length/2 - 0.3, -self.truck_width/2, front_camera_z), (0, 0, math.pi/6), "Front_Left"),
            ((-self.truck_length/2, 0, front_camera_z), (0, 0, math.pi), "Rear_Center"),
            ((0, self.truck_width/2, front_camera_z), (0, 0, -math.pi/2), "Side_Right"),
            ((0, -self.truck_width/2, front_camera_z), (0, 0, math.pi/2), "Side_Left"),
        ]

        for loc, rot, name in cameras:
            cam = self.create_camera_sensor(loc, rot, name)
            cam.parent = cab

        # LiDAR on roof
        lidar = self.create_lidar_sensor((0, 0, self.truck_height + 0.3), "Roof")
        lidar.parent = cab

    def create_lighting(self):
        """Add lighting to scene"""
        # Sun light
        bpy.ops.object.light_add(type='SUN', location=(10, 10, 20))
        sun = bpy.context.active_object
        sun.name = "Sun"
        sun.data.energy = 3.0

        # Area lights for better visualization
        bpy.ops.object.light_add(type='AREA', location=(5, 0, 10))
        area_light = bpy.context.active_object
        area_light.name = "Area_Light"
        area_light.data.energy = 500
        area_light.data.size = 10

    def create_camera(self):
        """Add render camera to scene"""
        bpy.ops.object.camera_add(location=(15, -15, 8))
        camera = bpy.context.active_object
        camera.name = "Render_Camera"

        # Point camera at origin
        direction = mathutils.Vector((0, 0, 0)) - camera.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        camera.rotation_euler = rot_quat.to_euler()

        # Set as active camera
        bpy.context.scene.camera = camera

    def create_ground_plane(self):
        """Create ground plane"""
        bpy.ops.mesh.primitive_plane_add(size=100, location=(0, 0, 0))
        ground = bpy.context.active_object
        ground.name = "Ground"

        # Add material
        mat = self.create_material("Ground_Material", (0.3, 0.3, 0.3, 1.0))
        ground.data.materials.append(mat)

    def generate_complete_truck(self):
        """Generate complete truck model with all components"""
        print("Generating truck cab...")
        cab = self.create_truck_cab()

        print("Adding windshield...")
        self.create_windshield(cab)

        print("Creating wheels...")
        self.create_truck_wheels(cab)

        print("Generating trailer...")
        trailer = self.create_trailer()
        self.create_trailer_wheels(trailer)

        print("Adding fifth wheel coupling...")
        self.create_fifth_wheel(cab, trailer)

        print("Adding sensors...")
        self.add_sensors(cab)

        print("Setting up scene...")
        self.create_ground_plane()
        self.create_lighting()
        self.create_camera()

        # Create collection for organization
        truck_collection = bpy.data.collections.new("Autonomous_Truck")
        bpy.context.scene.collection.children.link(truck_collection)

        # Move truck components to collection
        for obj in [cab, trailer]:
            if obj.name in bpy.context.scene.collection.objects:
                bpy.context.scene.collection.objects.unlink(obj)
                truck_collection.objects.link(obj)

        print("Truck model generation complete!")

        return cab, trailer

    def export_model(self, filepath: str, format: str = 'FBX'):
        """Export truck model"""
        if format == 'FBX':
            bpy.ops.export_scene.fbx(filepath=filepath)
        elif format == 'OBJ':
            bpy.ops.export_scene.obj(filepath=filepath)
        elif format == 'GLTF':
            bpy.ops.export_scene.gltf(filepath=filepath)

        print(f"Model exported to: {filepath}")

    def setup_animation(self, cab: bpy.types.Object, trailer: bpy.types.Object):
        """Setup basic animation for demonstration"""
        # Set frame range
        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = 250

        # Animate truck moving forward
        cab.location = (0, 0, 0)
        cab.keyframe_insert(data_path="location", frame=1)

        cab.location = (50, 0, 0)
        cab.keyframe_insert(data_path="location", frame=250)

        # Animate slight steering
        cab.rotation_euler[2] = 0
        cab.keyframe_insert(data_path="rotation_euler", frame=1)

        cab.rotation_euler[2] = 0.1
        cab.keyframe_insert(data_path="rotation_euler", frame=125)

        cab.rotation_euler[2] = 0
        cab.keyframe_insert(data_path="rotation_euler", frame=250)

        print("Animation setup complete!")


def main():
    """Main execution function"""
    print("="*60)
    print("Self-Driving Truck 3D Model Generator")
    print("="*60)

    generator = TruckModelGenerator()
    cab, trailer = generator.generate_complete_truck()

    # Optional: Setup animation
    print("\nSetting up animation...")
    generator.setup_animation(cab, trailer)

    # Optional: Export model
    # generator.export_model("/tmp/autonomous_truck.fbx", format='FBX')

    print("\n" + "="*60)
    print("Generation complete! Open Blender to view the model.")
    print("="*60)


if __name__ == "__main__":
    main()
