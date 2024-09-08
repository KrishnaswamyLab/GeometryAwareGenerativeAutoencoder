import bpy
import numpy as np
import os
import logging

# Function to delete all objects in the scene except the camera
def delete_all_objects_except_camera():
    for obj in bpy.context.scene.objects:
        if obj.type != 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)

def delete_lights():
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)

def load_trajectory(rpath='./'):
    methods = ['density', 'no_density', 'ours']
    datas = ['ellipsoid', 'torus', 'saddle', 'hemisphere']
    file_path = os.path.join(rpath, 'geodesic', 'geodesics_points')

    all_data = {}
    for dir_name in os.listdir(file_path):
        data_name = dir_name.split('_')[0]
        if data_name not in datas:
            continue
        if data_name not in all_data:
            all_data[data_name] = {}
        for method in methods:
            cur_datafile = np.load(f'{file_path}/{dir_name}/{method}.npz')
            all_data[data_name][method] = {}
            #print(cur_datafile.files)
            all_data[data_name][method]['x0'] = cur_datafile['x0']
            all_data[data_name][method]['x1'] = cur_datafile['x1']
            all_data[data_name][method]['xhat'] = cur_datafile['xhat'] # trajectory points [t, n, dim]

            all_data[data_name][method]['x'] = np.load(f'{rpath}/gt/{dir_name}.npz')['X']
            all_data[data_name][method]['geodesics'] = np.load(f'{rpath}/gt/{dir_name}.npz')['geodesics'] # [n, t, dim]
            all_data[data_name][method]['geodesics'] = np.transpose(all_data[data_name][method]['geodesics'], (1, 0, 2))
    
    print(all_data)
    return all_data


def create_mesh_surface(data_name="saddle", name="Mesh"):
    # Ensure the "Extra Objects" add-on is enabled
    bpy.ops.preferences.addon_enable(module='add_mesh_extra_objects')

    if data_name == 'saddle':
        eqt = "x**2 - y**2"
        x_range = 2
        y_range = 2
        # Add a "Z Surface" math mesh object
        bpy.ops.mesh.primitive_z_function_surface(
            equation=eqt,  # Example equation for the Z surface
            div_x=32,             # Number of divisions in X direction
            div_y=32,             # Number of divisions in Y direction
            size_x=x_range,          # Size in X direction
            size_y=y_range,          # Size in Y direction
        )
    elif data_name == 'ellipsoid':
        # Define the mathematical functions for X, Y, and Z
        x_function = "np.sin(u) * np.cos(v) * 3"
        y_function = "np.sin(u) * np.sin(v) * 2"
        z_function = "np.cos(u) * 1"

        # Add an XYZ Math Surface
        bpy.ops.mesh.primitive_xyz_function_surface(
            x_eq=x_function,
            y_eq=y_function,
            z_eq=z_function,
            range_u_min=-3.14,  # Min value for u
            range_u_max=3.14,   # Max value for u
            range_v_min=-6.28,  # Min value for v
            range_v_max=6.28,   # Max value for v
            range_u_step=128,           # Number of divisions in the U direction
            range_v_step=128,           # Number of divisions in the V direction
        )

    # Optional: Set the location of the mesh
    bpy.context.object.location = (0, 0, 0)
    # Set smooth shading
    bpy.ops.object.shade_smooth()
    
    mesh = bpy.context.object
    mesh.name = name
    
    # Create a semi-transparent material
    mat = bpy.data.materials.new(name="SemiTransparentMaterial")
    mat.use_nodes = True
    principled_bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if principled_bsdf:
        principled_bsdf.inputs["Base Color"].default_value = (0, 1, 1, 1)
        principled_bsdf.inputs["Alpha"].default_value = 0.8  # Semi-transparent

    # Enable transparency in material settings
    mat.blend_method = 'BLEND'
    mat.show_transparent_back = False

    # Assign material to the imported mesh
    if mesh.data.materials:
        mesh.data.materials[0] = mat
    else:
        mesh.data.materials.append(mat)
    

        
    return mesh
    

# Create Mesh from ply file
def create_mesh_from_ply(ply_file, name="Mesh"):
    bpy.ops.wm.ply_import(filepath=ply_file)
    mesh = bpy.context.object
    mesh.name = name
    
    # Set smooth shading
    bpy.ops.object.shade_smooth()
    
    # Create a semi-transparent material
    mat = bpy.data.materials.new(name="SemiTransparentMaterial")
    mat.use_nodes = True
    principled_bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if principled_bsdf:
        principled_bsdf.inputs["Base Color"].default_value = (0, 1, 1, 1)
        principled_bsdf.inputs["Alpha"].default_value = 0.8  # Semi-transparent

    # Enable transparency in material settings
    mat.blend_method = 'BLEND'
    mat.show_transparent_back = False

    # Assign material to the imported mesh
    if mesh.data.materials:
        mesh.data.materials[0] = mat
    else:
        mesh.data.materials.append(mat)

    return mesh

# Function to create a material for Grease Pencil strokes
def create_gpencil_material(gpencil_obj, name, color):
    material = bpy.data.materials.new(name=name)
    bpy.data.materials.create_gpencil_data(material)
    material.grease_pencil.color = color
    gpencil_obj.data.materials.append(material)
    
    return material

# Function to create Grease Pencil strokes from trajectory points with thickness
def create_gpencil_from_points(trajectories, gpencil_name="TrajectoryGPencil", 
                                layer_name="Trajectories", 
                                thickness=50, color=(1, 1, 0, 1)):
    # Create a new Grease Pencil object
    gpencil_data = bpy.data.grease_pencils.new(gpencil_name)
    gpencil_obj = bpy.data.objects.new(gpencil_name, gpencil_data)
    bpy.context.collection.objects.link(gpencil_obj)
    
    # Create a new Grease Pencil layer
    gp_layer = gpencil_data.layers.new(name=layer_name, set_active=True)
    
    # Create strokes in the Grease Pencil layer
    for traj in trajectories:
        frame = gp_layer.frames.new(0)  # Create a new frame at frame 0
        stroke = frame.strokes.new()
        stroke.display_mode = '3DSPACE'
        stroke.points.add(count=len(traj))
        #stroke.thickness = thickness  # Set the stroke thickness
        for i, point in enumerate(traj):
            stroke.points[i].co = point
            stroke.points[i].pressure = 1  # Set the stroke thickness
        stroke.line_width = thickness  # Set the stroke thickness directly

        
        # Set material for the stroke
        #color = colors[i % len(colors)]
        #material_name = f"GPencilMaterial_{i}"
        #material = create_gpencil_material(gpencil_obj, material_name, color)
        #stroke.material_index = gpencil_obj.data.materials.find(material_name)

        gpencil_data.pixel_factor = thickness

    return gpencil_obj

# Function to create a curve from trajectory points
def create_curve_from_points(trajectories, name="TrajectoryCurve", width=.01):
    curve_data = bpy.data.curves.new(name=name, type='CURVE')
    curve_data.dimensions = '3D'
    
    for traj in trajectories:
        spline = curve_data.splines.new('BEZIER')
        spline.bezier_points.add(len(traj) - 1)
        for i, point in enumerate(traj):
            bp = spline.bezier_points[i]
            bp.co = point
            bp.handle_left_type = bp.handle_right_type = 'AUTO'
    
    # Increase width of the curve
    bevel_object_name = f"BevelObject_{name}"
    bpy.ops.mesh.primitive_circle_add(radius=width, vertices=16, fill_type='NGON', location=(0, 0, 0))
    bevel_object = bpy.context.object
    bevel_object.name = bevel_object_name
    
    curve_data.bevel_object = bevel_object

    curve_obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(curve_obj)
    curve_obj.data.bevel_depth = width
    
    return curve_obj

# Function to create points
def create_point(location, name="Point", radius=0.05, color=(1, 0, 0, 1)):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location)
    point = bpy.context.object
    point.name = name
    mat = bpy.data.materials.new(name="PointMaterial")
    mat.diffuse_color = color
    if point.data.materials:
        point.data.materials[0] = mat
    else:
        point.data.materials.append(mat)
    return point

# Function to create multiple lights around the torus
def create_lights_around_torus(torus_center, distance, count=4):
    for i in range(count):
        angle = 2 * np.pi * i / count
        x = torus_center[0] + distance * np.cos(angle)
        y = torus_center[1] + distance * np.sin(angle)
        z = torus_center[2] - np.pi
        bpy.ops.object.light_add(type='POINT', location=(x, y, z))
        light = bpy.context.object
        light.data.energy = 500

def create_scene():
    # ==== Create the mesh, trajectories, and points in Blender ====
    methods = ['gt', 'density', 'no_density', 'ours']
    datas = ['torus', 'ellipsoid', 'saddle', 'hemisphere']
    rp = '/Users/danqiliao/Desktop/dmae/src/comparison/'
    methods = 'ours'
    data_name = 'ellipsoid'

    # Load the trajectory data.
    all_data = load_trajectory(rp)
    print(all_data)
        
    # Create the mesh.
    if data_name == 'torus':
        create_mesh_from_ply(f"{rp}/pointclouds/{data_name}_mesh.ply")
    else:
        create_mesh_surface(data_name=data_name)

    # Create trajectories.
    #traj_index = np.arange(20)
    traj_index = [19,10,13,12,16]
    if data_name == 'torus':
        traj_index = [19,10,13,12] #TORUS; t10, r0.1
        thickness = 10
        pradius = 0.1
        ldist = 5
    elif data_name == 'saddle':
        traj_index = [3,4,8,14] # SADDLE; t6; r0.05, l2.5
        thickness = 6
        pradius = 0.05
        ldist = 2.5
        
    if methods == 'gt':
        example = all_data[data_name]['density']['geodesics'] # [t, n, dim]
    else:
        example = all_data[data_name][methods]['xhat']
    trajectories = np.transpose(example, (1, 0, 2)) # [n, t, dim]
    for i, traj in enumerate(trajectories[traj_index]):
        create_gpencil_from_points([traj], gpencil_name=f"Trajectory_{i}", 
        layer_name="Trajectories", thickness=thickness)
        

    ## Create start and end points.
    start_points = all_data[data_name]['density']['x0']
    end_points = all_data[data_name]['density']['x1']
    # Create start and end points in Blender
    for i, loc in enumerate(start_points[traj_index]):
        create_point(location=loc, name=f"StartPoint_{i}", color=(1, 0, 0, 1))

    for i, loc in enumerate(end_points[traj_index]):
        create_point(location=loc, name=f"EndPoint_{i}", color=(0, 0, 1, 1))
        
    ## LIGHTS.
    center = (0, 0, 0)  # Assuming the torus is centered at the origin
    create_lights_around_torus(center, distance=ldist, count = 4)


create_scene()

## LIGHTS.
#center = (0, 0, 0)  # Assuming the torus is centered at the origin
#create_lights_around_torus(center, distance=2.5, count = 4)

delete_lights()
delete_all_objects_except_camera()


# Enable transparency for the render
#bpy.context.scene.render.film_transparent = True

## Enable compositing
#bpy.context.scene.use_nodes = True
#tree = bpy.context.scene.node_tree
#nodes = tree.nodes
#links = tree.links

## Clear default nodes
#for node in nodes:
#    nodes.remove(node)

## Add nodes
#render_layers = nodes.new(type='CompositorNodeRLayers')
#composite = nodes.new(type='CompositorNodeComposite')
#alpha_over = nodes.new(type='CompositorNodeAlphaOver')
#rgb = nodes.new(type='CompositorNodeRGB')
#viewer = nodes.new(type='CompositorNodeViewer')  # Optional, for preview in the compositor

## Set the RGB node to white
#rgb.outputs[0].default_value = (1, 1, 1, 1)  # White color

## Set the alpha over factor to 1 for full opacity
#alpha_over.inputs['Fac'].default_value = 1.0

## Connect nodes
#links.new(render_layers.outputs['Image'], alpha_over.inputs[2])  # Render layers to bottom input of Alpha Over
#links.new(rgb.outputs[0], alpha_over.inputs[1])  # White color to top input of Alpha Over
#links.new(alpha_over.outputs[0], composite.inputs[0])  # Alpha Over output to Composite node
#links.new(alpha_over.outputs[0], viewer.inputs[0])  # Optional, for preview