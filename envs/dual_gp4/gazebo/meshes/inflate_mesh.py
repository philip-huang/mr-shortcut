import trimesh
import numpy as np

def inflate_mesh(input_file, output_file, inflation_amount=0.01):
    # Load the mesh
    mesh = trimesh.load_mesh(input_file)
    
    # Calculate vertex normals
    vertex_normals = mesh.vertex_normals
    
    # Inflate vertices along their normals
    new_vertices = mesh.vertices + (vertex_normals * inflation_amount)
    
    # Create new mesh with inflated vertices
    inflated_mesh = trimesh.Trimesh(
        vertices=new_vertices,
        faces=mesh.faces,
        process=False
    )
    
    # Export the inflated mesh
    inflated_mesh.export(output_file)

# Example usage
input_file = "eoat/D405_mount.stl"
output_file = "eoat/D405_mount_inflated.stl"
inflate_mesh(input_file, output_file, inflation_amount=0.01)