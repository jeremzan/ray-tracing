from helper_classes import *
import matplotlib.pyplot as plt
import numpy as np


def render_scene(camera, ambient, lights, objects, screen_size, max_depth):
    width, height = screen_size
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

    image = np.zeros((height, width, 3))

    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            # screen is on origin
            pixel = np.array([x, y, 0])
            origin = camera
            direction = normalize(pixel - origin)
            ray = Ray(origin, direction)
            color = np.zeros(3)
            hit_object, min_distance = ray.nearest_intersected_object(objects)
            if hit_object is None:
                continue
            hit_point = ray.origin + ray.direction * min_distance
            offset_point = hit_point +  1e-2 * hit_object.get_normal(hit_point)
            color = get_color(origin, ambient, lights, objects, hit_object, ray, offset_point, 0, max_depth)

            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color, 0, 1)

    return image


def calc_ambient_color(ambient, hit_object):
    return ambient * hit_object.ambient



def calc_diffuse(hit_object, object_normal, light_direction, light_intensity):
    return light_intensity * hit_object.diffuse * np.dot(object_normal, light_direction)

def calc_specular(hit_object, light_intensity, view_direction, reflected_direction):
    return light_intensity * hit_object.specular * (np.dot(view_direction, reflected_direction) ** hit_object.shininess)
    

def get_color(origin, ambient, lights, objects, hit_object, ray, hit_point, depth, max_depth):
    color = np.zeros(3)
    color += calc_ambient_color(ambient, hit_object)

    for light in lights:
        light_ray = light.get_light_ray(hit_point)
        _ , shadow_dist = light_ray.nearest_intersected_object(objects)
        if shadow_dist < light.get_distance_from_light(hit_point):
            color = np.zeros(3)
            continue

        light_dir = light.get_light_ray(hit_point).direction
        light_intensity = light.get_intensity(hit_point)
        view_dir = normalize(origin - hit_point)
        reflected_dir = reflected(-light_dir, hit_object.get_normal(hit_point))
        
        color += calc_diffuse(hit_object, hit_object.get_normal(hit_point), light_dir, light_intensity)
        color += calc_specular(hit_object, light_intensity, view_dir, reflected_dir)

    if depth + 1 < max_depth:
        reflected_dir = reflected(ray.direction, hit_object.get_normal(hit_point))
        reflected_ray = Ray(hit_point, reflected_dir)
        new_hit_object, distance = reflected_ray.nearest_intersected_object(objects)
        if new_hit_object is not None:
            new_hit_point = reflected_ray.origin + reflected_ray.direction * distance
            offset_point = new_hit_point + 1e-2 * new_hit_object.get_normal(new_hit_point)
            reflected_color = get_color(origin, ambient, lights, objects, new_hit_object, reflected_ray, offset_point, depth+1, max_depth)
            color += reflected_color * hit_object.reflection 
    return color



# Write your own objects and lights
# TODO
def your_own_scene():
    camera = np.array([0,0,1])
    lights = []
    objects = []
    return camera, lights, objects