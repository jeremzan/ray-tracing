from helper_classes import *
import numpy as np
import matplotlib.pyplot as plt


def render_scene(camera, ambient, lights, objects, screen_size, max_depth):
    width, height = screen_size
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

    image = np.zeros((height, width, 3))
    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            pixel = np.array([x, y, 0])
            direction = normalize(pixel - camera)
            ray = Ray(camera, direction)
            hit_object, min_distance = ray.nearest_intersected_object(objects)
            if hit_object is not None:
                hit_point = ray.origin + ray.direction * min_distance
                color = get_color(hit_object, hit_point, ray, objects, lights, ambient, 0, max_depth)
            else:
                color = np.zeros(3)
            image[i, j] = np.clip(color, 0, 1)

    return image



def calc_ambient(material_ambient, global_ambient):
    return material_ambient * global_ambient

def calc_diffuse(material_diffuse, light_intensity, normal, light_dir):
    return material_diffuse * light_intensity * max(np.dot(normal, light_dir), 0)

def calc_specular(material_specular, light_intensity, view_dir, normal, light_dir, shininess):
    reflect_dir = reflected(-light_dir, normal)
    return material_specular * light_intensity * max(np.dot(view_dir, reflect_dir), 0) ** shininess


def get_color(hit_object, hit_point, ray, objects, lights, ambient, depth, max_depth, epsilon=1e-5):
    if hit_object is None:
        return np.zeros(3)

    normal = hit_object.normal if hasattr(hit_object, 'normal') else normalize(hit_point - hit_object.center)
    view_dir = normalize(-ray.direction)

    # Apply epsilon offset above the surface
    offset_point = hit_point + normalize(normal) * epsilon

    color = calc_ambient(hit_object.ambient, ambient)

    for light in lights:
        light_ray = light.get_light_ray(offset_point)
        shadow_obj, shadow_dist = light_ray.nearest_intersected_object(objects)
        if shadow_obj is None or shadow_dist > light.get_distance_from_light(offset_point):
            light_intensity = light.get_intensity(offset_point)
            light_dir = normalize(light_ray.direction)
            color += calc_diffuse(hit_object.diffuse, light_intensity, normal, light_dir)
            color += calc_specular(hit_object.specular, light_intensity, view_dir, normal, light_dir, hit_object.shininess)

    if hit_object.reflection > 0 and depth + 1 < max_depth:
        reflect_dir = reflected(ray.direction, normal)
        reflect_ray = Ray(offset_point, reflect_dir)
        nearest_reflected_obj, nearest_reflected_dist = reflect_ray.nearest_intersected_object(objects)
        reflected_hit_point = reflect_ray.origin + reflect_ray.direction * nearest_reflected_dist
        reflected_color = get_color(nearest_reflected_obj, reflected_hit_point, reflect_ray, objects, lights, ambient, depth + 1, max_depth)
        color += reflected_color * hit_object.reflection

    return color



# Write your own objects and lights
# TODO
def your_own_scene():
    camera = np.array([0,0,1])
    lights = []
    objects = []
    return camera, lights, objects
