"""
Blender Script: Environment and Road Generator

Creates realistic road environments for autonomous truck simulation including
highways, urban roads, traffic, and obstacles.

Usage:
    blender --background --python environment.py

Author: Self-Driving Truck Project
Date: 2025-11-17
"""

import bpy
import math
import mathutils
import random
from typing import List, Tuple


class EnvironmentGenerator:
    """Generate road environments for simulation"""

    def __init__(self):
        self.lane_width = 3.5  # meters
        self.road_segments = []

    def create_material(self, name: str, color: Tuple[float, float, float, float]):
        """Create material with color"""
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()

        output = nodes.new('ShaderNodeOutputMaterial')
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.inputs['Base Color'].default_value = color
        bsdf.inputs['Roughness'].default_value = 0.8

        mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        return mat

    def create_highway_straight(self, length: float = 1000, num_lanes: int = 3):
        """Create straight highway section"""
        road_width = self.lane_width * num_lanes * 2  # Both directions

        # Main road surface
        bpy.ops.mesh.primitive_cube_add(
            size=1,
            location=(length/2, 0, -0.1)
        )
        road = bpy.context.active_object
        road.name = "Highway_Road"
        road.scale = (length, road_width, 0.2)

        # Apply asphalt material
        asphalt_mat = self.create_material("Asphalt", (0.1, 0.1, 0.1, 1.0))
        road.data.materials.append(asphalt_mat)

        # Lane markings
        self.create_lane_markings(length, num_lanes)

        # Road shoulders
        self.create_shoulders(length, road_width)

        # Barriers
        self.create_barriers(length, road_width)

        return road

    def create_lane_markings(self, length: float, num_lanes: int):
        """Create lane marking lines"""
        marking_mat = self.create_material("Lane_Marking", (1.0, 1.0, 1.0, 1.0))

        # Center divider (solid yellow)
        center_mat = self.create_material("Center_Divider", (1.0, 0.9, 0.0, 1.0))

        # Dashed lines
        dash_length = 3.0
        gap_length = 9.0
        segment_length = dash_length + gap_length

        num_segments = int(length / segment_length)

        for lane in range(-num_lanes + 1, num_lanes):
            if lane == 0:
                # Solid center divider
                bpy.ops.mesh.primitive_cube_add(
                    size=1,
                    location=(length/2, 0, 0.01)
                )
                divider = bpy.context.active_object
                divider.scale = (length, 0.15, 0.02)
                divider.data.materials.append(center_mat)
            else:
                # Dashed lane markings
                y_pos = lane * self.lane_width

                for i in range(num_segments):
                    x_pos = i * segment_length + dash_length/2

                    bpy.ops.mesh.primitive_cube_add(
                        size=1,
                        location=(x_pos, y_pos, 0.01)
                    )
                    dash = bpy.context.active_object
                    dash.scale = (dash_length, 0.1, 0.02)
                    dash.data.materials.append(marking_mat)

    def create_shoulders(self, length: float, road_width: float):
        """Create road shoulders"""
        shoulder_width = 2.5
        shoulder_mat = self.create_material("Shoulder", (0.2, 0.15, 0.1, 1.0))

        for side in [-1, 1]:
            y_pos = side * (road_width/2 + shoulder_width/2)

            bpy.ops.mesh.primitive_cube_add(
                size=1,
                location=(length/2, y_pos, -0.15)
            )
            shoulder = bpy.context.active_object
            shoulder.scale = (length, shoulder_width, 0.1)
            shoulder.data.materials.append(shoulder_mat)

    def create_barriers(self, length: float, road_width: float):
        """Create highway barriers"""
        barrier_mat = self.create_material("Barrier", (0.8, 0.8, 0.8, 1.0))
        barrier_height = 1.0
        barrier_width = 0.3

        spacing = 5.0
        num_barriers = int(length / spacing)

        for side in [-1, 1]:
            y_pos = side * (road_width/2 + 3.0)

            for i in range(num_barriers):
                x_pos = i * spacing

                bpy.ops.mesh.primitive_cube_add(
                    size=1,
                    location=(x_pos, y_pos, barrier_height/2)
                )
                barrier = bpy.context.active_object
                barrier.scale = (spacing * 0.9, barrier_width, barrier_height)
                barrier.data.materials.append(barrier_mat)

    def create_curved_road(self, radius: float = 100, angle_degrees: float = 90,
                          num_lanes: int = 3):
        """Create curved highway section"""
        angle_rad = math.radians(angle_degrees)
        segments = 20
        road_width = self.lane_width * num_lanes * 2

        vertices = []
        faces = []

        # Generate vertices for road mesh
        for i in range(segments + 1):
            theta = (i / segments) * angle_rad
            inner_r = radius - road_width/2
            outer_r = radius + road_width/2

            # Inner edge
            x_inner = inner_r * math.cos(theta)
            y_inner = inner_r * math.sin(theta)
            vertices.append((x_inner, y_inner, 0))

            # Outer edge
            x_outer = outer_r * math.cos(theta)
            y_outer = outer_r * math.sin(theta)
            vertices.append((x_outer, y_outer, 0))

        # Create faces
        for i in range(segments):
            v1 = i * 2
            v2 = i * 2 + 1
            v3 = i * 2 + 3
            v4 = i * 2 + 2
            faces.append((v1, v2, v3, v4))

        # Create mesh
        mesh = bpy.data.meshes.new("Curved_Road_Mesh")
        mesh.from_pydata(vertices, [], faces)
        mesh.update()

        # Create object
        curved_road = bpy.data.objects.new("Curved_Road", mesh)
        bpy.context.collection.objects.link(curved_road)

        # Add material
        asphalt_mat = self.create_material("Curved_Asphalt", (0.1, 0.1, 0.1, 1.0))
        curved_road.data.materials.append(asphalt_mat)

        return curved_road

    def create_intersection(self, size: float = 30):
        """Create intersection"""
        # Main intersection area
        bpy.ops.mesh.primitive_cube_add(
            size=1,
            location=(0, 0, -0.1)
        )
        intersection = bpy.context.active_object
        intersection.name = "Intersection"
        intersection.scale = (size, size, 0.2)

        asphalt_mat = self.create_material("Intersection_Asphalt", (0.12, 0.12, 0.12, 1.0))
        intersection.data.materials.append(asphalt_mat)

        # Crosswalks
        self.create_crosswalks(size)

        # Traffic lights
        self.create_traffic_lights(size)

        return intersection

    def create_crosswalks(self, intersection_size: float):
        """Create crosswalk markings"""
        crosswalk_mat = self.create_material("Crosswalk", (1.0, 1.0, 1.0, 1.0))

        stripe_width = 0.5
        stripe_spacing = 0.3
        crosswalk_width = 4.0
        num_stripes = int(crosswalk_width / (stripe_width + stripe_spacing))

        # Four crosswalks (one on each side)
        positions = [
            (intersection_size/2 + 2, 0, 0, 0),  # East
            (-intersection_size/2 - 2, 0, 0, 0),  # West
            (0, intersection_size/2 + 2, math.pi/2, 0),  # North
            (0, -intersection_size/2 - 2, math.pi/2, 0),  # South
        ]

        for x, y, rot_z, _ in positions:
            for i in range(num_stripes):
                offset = (i - num_stripes/2) * (stripe_width + stripe_spacing)

                bpy.ops.mesh.primitive_cube_add(
                    size=1,
                    location=(x, y + offset, 0.01)
                )
                stripe = bpy.context.active_object
                stripe.scale = (crosswalk_width, stripe_width, 0.02)
                stripe.rotation_euler = (0, 0, rot_z)
                stripe.data.materials.append(crosswalk_mat)

    def create_traffic_lights(self, intersection_size: float):
        """Create traffic light poles"""
        pole_height = 5.0
        offset = intersection_size/2 + 1

        positions = [
            (offset, offset),
            (offset, -offset),
            (-offset, offset),
            (-offset, -offset),
        ]

        for x, y in positions:
            # Pole
            bpy.ops.mesh.primitive_cylinder_add(
                radius=0.15,
                depth=pole_height,
                location=(x, y, pole_height/2)
            )
            pole = bpy.context.active_object

            pole_mat = self.create_material(f"Pole_{x}_{y}", (0.2, 0.2, 0.2, 1.0))
            pole.data.materials.append(pole_mat)

            # Light housing
            bpy.ops.mesh.primitive_cube_add(
                size=1,
                location=(x, y, pole_height + 0.5)
            )
            housing = bpy.context.active_object
            housing.scale = (0.3, 0.8, 0.3)

            # Lights (red, yellow, green)
            colors = [
                (1.0, 0.0, 0.0, 1.0),  # Red
                (1.0, 1.0, 0.0, 1.0),  # Yellow
                (0.0, 1.0, 0.0, 1.0),  # Green
            ]

            for i, color in enumerate(colors):
                light_y = y + (i - 1) * 0.25

                bpy.ops.mesh.primitive_uv_sphere_add(
                    radius=0.1,
                    location=(x, light_y, pole_height + 0.5)
                )
                light = bpy.context.active_object

                light_mat = self.create_material(f"Light_{x}_{y}_{i}", color)
                light_mat.use_nodes = True
                emission = light_mat.node_tree.nodes.new('ShaderNodeEmission')
                emission.inputs['Color'].default_value = color
                emission.inputs['Strength'].default_value = 2.0

                light.data.materials.append(light_mat)

    def create_vehicle_obstacle(self, location: Tuple[float, float, float],
                               vehicle_type: str = 'car'):
        """Create simple vehicle obstacle"""
        if vehicle_type == 'car':
            length, width, height = 4.5, 2.0, 1.5
            color = (random.random(), random.random(), random.random(), 1.0)
        elif vehicle_type == 'truck':
            length, width, height = 7.0, 2.5, 3.0
            color = (0.2, 0.2, 0.8, 1.0)
        else:
            length, width, height = 2.0, 0.8, 1.2
            color = (0.9, 0.9, 0.1, 1.0)

        bpy.ops.mesh.primitive_cube_add(
            size=1,
            location=location
        )
        vehicle = bpy.context.active_object
        vehicle.name = f"Vehicle_{vehicle_type}"
        vehicle.scale = (length, width, height)

        mat = self.create_material(f"Vehicle_Mat_{vehicle_type}", color)
        vehicle.data.materials.append(mat)

        # Add wheels
        wheel_positions = [
            (length/3, width/2, -height/3),
            (length/3, -width/2, -height/3),
            (-length/3, width/2, -height/3),
            (-length/3, -width/2, -height/3),
        ]

        wheel_mat = self.create_material("Wheel_Mat", (0.1, 0.1, 0.1, 1.0))

        for wx, wy, wz in wheel_positions:
            bpy.ops.mesh.primitive_cylinder_add(
                radius=0.3,
                depth=0.2,
                location=(location[0] + wx, location[1] + wy, location[2] + wz),
                rotation=(0, math.pi/2, 0)
            )
            wheel = bpy.context.active_object
            wheel.data.materials.append(wheel_mat)
            wheel.parent = vehicle

        return vehicle

    def populate_traffic(self, road_length: float, num_lanes: int, density: int = 10):
        """Populate road with traffic"""
        vehicles = []

        for _ in range(density):
            x = random.uniform(0, road_length)
            lane = random.randint(-num_lanes + 1, num_lanes - 1)
            y = lane * self.lane_width + random.uniform(-0.5, 0.5)
            z = 0.75

            vehicle_type = random.choice(['car', 'car', 'car', 'truck'])
            vehicle = self.create_vehicle_obstacle((x, y, z), vehicle_type)
            vehicles.append(vehicle)

        return vehicles

    def create_signs(self, positions: List[Tuple[float, float, str]]):
        """Create traffic signs"""
        for x, y, sign_type in positions:
            # Sign pole
            bpy.ops.mesh.primitive_cylinder_add(
                radius=0.05,
                depth=2.5,
                location=(x, y, 1.25)
            )
            pole = bpy.context.active_object

            # Sign board
            bpy.ops.mesh.primitive_cube_add(
                size=1,
                location=(x, y, 2.5)
            )
            sign = bpy.context.active_object
            sign.scale = (0.8, 0.05, 0.8)

            # Color based on type
            if 'speed' in sign_type.lower():
                color = (1.0, 0.0, 0.0, 1.0)
            elif 'warning' in sign_type.lower():
                color = (1.0, 1.0, 0.0, 1.0)
            else:
                color = (0.0, 0.0, 1.0, 1.0)

            mat = self.create_material(f"Sign_{sign_type}", color)
            sign.data.materials.append(mat)

    def create_complete_highway_scene(self):
        """Create complete highway scene with all elements"""
        print("Creating highway road...")
        road_length = 500
        num_lanes = 3
        road = self.create_highway_straight(road_length, num_lanes)

        print("Populating traffic...")
        vehicles = self.populate_traffic(road_length, num_lanes, density=15)

        print("Adding traffic signs...")
        sign_positions = [
            (50, num_lanes * self.lane_width + 5, 'speed_limit_100'),
            (150, num_lanes * self.lane_width + 5, 'warning_curve'),
            (300, num_lanes * self.lane_width + 5, 'speed_limit_80'),
        ]
        self.create_signs(sign_positions)

        print("Adding lighting...")
        bpy.ops.object.light_add(type='SUN', location=(0, 0, 100))
        sun = bpy.context.active_object
        sun.data.energy = 2.0

        print("Highway scene complete!")

        return road


def main():
    """Main execution"""
    print("="*60)
    print("Environment and Road Generator")
    print("="*60)

    # Clear scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    generator = EnvironmentGenerator()
    generator.create_complete_highway_scene()

    print("\n" + "="*60)
    print("Environment generation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
