from functools import wraps
from pathlib import Path
from typing import Union, List
import os
import json
from multiprocessing import Pool
from pprint import pprint

import numpy as np
import openfoamparser_mai as Ofpp
import pyvista

def save_json(filename, data, save_path) -> None:
    """Cохраняет json"""
    file_path = save_path / Path(filename)
    with open(file_path, 'w', encoding="utf8") as f:
        json.dump(data, f)


def save_json_in_chunks(filename, data, save_path, chunk_size=1000):
    full_path = os.path.join(save_path, filename)
    with open(full_path, 'w') as file:
        file.write('[')
        for i, item in enumerate(data):
            json_str = json.dumps(item)
            file.write(json_str)
            if i < len(data) - 1:
                file.write(',\n')
            if i % chunk_size == 0 and i != 0:
                file.flush()  # Flush data to disk periodically
        file.write(']')


# The wrapper function for multiprocessing
def save_json_in_chunks_wrapper(args):
    save_json_in_chunks(*args)


def json_streaming_writer(filepath, data_func, data_args):
    """Write JSON data to a file using a generator to minimize memory usage."""
    data_gen = data_func(*data_args)
    try:
        with open(filepath, 'w') as file:
            print(f"writing {filepath}")
            file.write('[')
            for i, item in enumerate(data_gen):
                if i != 0:  # Add a comma before all but the first item
                    file.write(',')
                json.dump(item, file)
            file.write(']')
        print(f"Finished writing {filepath}")
    except Exception as e:
        print(f"Failed to write {filepath}: {str(e)}") 


def create_nodes_gen(mesh_bin):
    """Generator for nodes."""
    for point in mesh_bin.points:
        yield {
            'X': point[0],
            'Y': point[1],
            'Z': point[2]
        }


def create_faces_gen(mesh_bin):
    """Generator for faces."""
    for face in mesh_bin.faces:
        yield list(face)


def create_elements_gen(mesh_bin, p, u, c):
    """Generator for elements."""
    for i, cell in enumerate(mesh_bin.cell_faces):
        yield {
            'Faces': cell,
            'Pressure': p[i],
            'Velocity': {
                'X': u[i][0],
                'Y': u[i][1],
                'Z': u[i][2]
            },
            'VelocityModule': np.linalg.norm(u[i]),
            'Position': {
                'X': c[i][0],
                'Y': c[i][1],
                'Z': c[i][2]
            }
        }


def create_surfaces_gen(surfaces):
    """Generator for surfaces."""
    for surface in surfaces:
        yield surface


def _face_center_position(points: list, mesh: Ofpp.FoamMesh) -> list:
    vertecis = [mesh.points[p] for p in points]
    vertecis = np.array(vertecis)
    return list(vertecis.mean(axis=0))



def process_computational_domain(solver_path: Union[str, os.PathLike, Path],
                                 save_path: Union[str, os.PathLike, Path],
                                 p: np.ndarray,
                                 u: np.ndarray,
                                 c: np.ndarray,
                                 surface_name: str, 
                                 input_domain_name: str = 'motorBike_0') -> None:
    """Сохранение геометрии расчетной области в виде json файла с полями:
    'Nodes' - List[x: float, y: float, z:float], 
    'Faces' - List [List[int]], 
    'Elements' - List [Dict{Faces: List[int],
                            Pressure: float,
                            Velocity: List[float],
                            VelocityModule: float,
                            Position: List[float]}
                            ], 
    'Surfaces' - List[
                    Tuple[Surface_name: str, 
                    List[Dict{ParentElementID: int,
                              ParentFaceId: int,
                              Position: List[float]}]
                    ]

    Args:
        solver_path (Union[str, os.PathLike, Path]): Путь до папки с расчетом.
        save_path (Union[str, os.PathLike, Path]): Путь для сохранения итогового json.
        p (np.ndarray): Поле давления.
        u (np.ndarray): Поле скоростей.
        c (np.ndarray): Центры ячеек.
        surface_name (str): Имя для поверхности.
    """
    
    # Step 0: parse mesh and scale vertices
    mesh_bin = Ofpp.FoamMesh(solver_path )

    # Step I: compute TFemFace_Surface
    domain_names = [input_domain_name.encode('ascii')]
    surfaces = []

    for i, domain_name in enumerate(domain_names):
        bound_cells = list(mesh_bin.boundary_cells(domain_name))

        boundary_faces = []
        boundary_faces_cell_ids = []
        for bc_id in bound_cells:
            faces = mesh_bin.cell_faces[bc_id]
            for f in faces:
                if mesh_bin.is_face_on_boundary(f, domain_name):
                    boundary_faces.append(f)
                    boundary_faces_cell_ids.append(bc_id)

        f_b_set = set(zip(boundary_faces, boundary_faces_cell_ids))

        body_faces = []
        for f, b in f_b_set:
            try:
                position = _face_center_position(mesh_bin.faces[f], mesh_bin)
                d = {'ParentElementID': b,
                    'ParentFaceId': f,
                    'Position': {'X': position[0], 'Y': position[1], 'Z': position[2]}
                    }
                body_faces.append(d)
            except IndexError:
                print(f'Indexes for points: {f} is not valid!')

        surfaces.append({'Item1': surface_name,
                'Item2': body_faces}) 
    
    # Define file paths
    nodes_path = os.path.join(save_path, 'Nodes.json')
    faces_path = os.path.join(save_path, 'Faces.json')
    elements_path = os.path.join(save_path, 'Elements.json')
    surfaces_path = os.path.join(save_path, 'Surfaces.json')

    # Prepare arguments for the multiprocessing function
    
    tasks = [
    (nodes_path, create_nodes_gen, (mesh_bin,)),
    (faces_path, create_faces_gen, (mesh_bin,)),
    (elements_path, create_elements_gen, (mesh_bin, p, u, c)),
    (surfaces_path, create_surfaces_gen, (surfaces,))
        ]

    # Use multiprocessing pool
    with Pool() as pool:
        pool.starmap(json_streaming_writer, tasks)



def pressure_field_on_surface(solver_path: Union[str, os.PathLike, Path],
                                p: np.ndarray,
                                surface_name: str = 'Surface',
                                input_domain_name: str = 'motorBike_0'
                                 ) -> None:
    """Поле давлений на поверхности тела:
    'Nodes' - List[x: float, y: float, z:float], 
    'Faces' - List [List[int]], 
    'Elements' - List [Dict{Faces: List[int],
                            Pressure: float,
                            Velocity: List[float],
                            VelocityModule: float,
                            Position: List[float]}
                            ], 
    'Surfaces' - List[
                    Tuple[Surface_name: str, 
                    List[Dict{ParentElementID: int,
                              ParentFaceId: int,
                              Position: List[float]}]
                    ]

    Args:
        solver_path (Union[str, os.PathLike, Path]): Путь до папки с расчетом.
        p (np.ndarray): Поле давления.
        surface_name (str): Имя для поверхности.
    """
    
    # Step 0: parse mesh and scale vertices
    mesh_bin = Ofpp.FoamMesh(solver_path )

    # Step I: compute TFemFace_Surface
    domain_names = [input_domain_name.encode('ascii')]
    surfaces = []

    for i, domain_name in enumerate(domain_names):
        bound_cells = list(mesh_bin.boundary_cells(domain_name))

        boundary_faces = []
        boundary_faces_cell_ids = []
        for bc_id in bound_cells:
            faces = mesh_bin.cell_faces[bc_id]
            for f in faces:
                if mesh_bin.is_face_on_boundary(f, domain_name):
                    boundary_faces.append(f)
                    boundary_faces_cell_ids.append(bc_id)

        f_b_set = set(zip(boundary_faces, boundary_faces_cell_ids))

        body_faces = []
        for f, b in f_b_set:
            try:
                position = _face_center_position(mesh_bin.faces[f], mesh_bin)
                d = {'ParentElementID': b,
                    'ParentFaceId': f,
                    'CentrePosition': {'X': position[0], 'Y': position[1], 'Z': position[2]},
                    'PressureValue': p[b]
                    }
                body_faces.append(d)
            except IndexError:
                print(f'Indexes for points: {f} is not valid!')

        surfaces.append({'Item1': surface_name,
                'Item2': body_faces}) 
        

        return surfaces
    


def plot_slices(solver_path: Union[str, os.PathLike, Path],
                base_path: Union[str, os.PathLike, Path],
                show_edges: bool = False,
                body_name_of: str = 'motorBike_0',
                body_name_scene: str = 'Body') -> None:
    """Построение графиков с распределением давления на теле.

    Args:
        solver_path (Union[str, os.PathLike, Path]): Путь до папки с кейсом.
        base_path (Union[str, os.PathLike, Path]): Путь до папки с финальным временем расчета.
        show_edges (bool, optional): Нужно ли визуализировать расчетную область. Defaults to False.
        body_name_of (str, optional):  Имя тела в сцене (как в OF). Defaults to 'motorBike_0'.
        body_name_scene (str, optional):  Имя тела в сцене (как в config). Defaults to 'motorBike_0'.
    """
    # Step 1: prepare case for reading
    path_to_case = solver_path / Path('foam.foam')
    save_path_body = base_path / Path(f'p_on_surface_{body_name_scene}.png')
    
    if not os.path.exists(path_to_case):
        open(path_to_case, 'w').close()

    # Step 2: read case and make z slise
    reader = pyvista.POpenFOAMReader(path_to_case)

    block = reader.read()
    body = block['boundary'][body_name_of]


    # Step 3: plot pressure on body surface
    pl = pyvista.Plotter(off_screen=True)
    pl.add_mesh(body,
                scalars='p',
                lighting=False,
                scalar_bar_args={'title': 'Flow Preassure'},
                cmap = 'jet',
                show_edges=show_edges)

    pl.enable_anti_aliasing()
    pl.enable_parallel_projection()
    pl.show_axes()
    pl.camera.focal_point = body.center
    pl.camera.position = (-10, 10, -10)
    pl.camera.up = (0, 1, 0)

    _ = pl.screenshot(save_path_body)