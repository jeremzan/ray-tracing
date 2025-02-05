import numpy as np


# This function gets a vector and returns its normalized form.
def normalize(vector):
    return vector / np.linalg.norm(vector)


# This function gets a vector and the normal of the surface it hit
# This function returns the vector that reflects from the surface
def reflected(vector, axis):
    normalized_axis = normalize(axis)
    return vector - 2 * (np.dot(vector, normalized_axis) * normalized_axis)

## Lights


class LightSource:
    def __init__(self, intensity):
        self.intensity = intensity


class DirectionalLight(LightSource):

    def __init__(self, intensity, direction):
        super().__init__(intensity)
        self.direction = normalize(direction)

    # This function returns the ray that goes from a point to the light source
    def get_light_ray(self,intersection_point):
        return Ray(intersection_point,-self.direction)

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        return np.inf


    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        return self.intensity


class PointLight(LightSource):
    def __init__(self, intensity, position, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from a point to the light source
    def get_light_ray(self,intersection):
        return Ray(intersection, normalize(self.position - intersection))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self,intersection):
        return np.linalg.norm(intersection - self.position)

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        return self.intensity / (self.kc + self.kl*d + self.kq * (d**2))


class SpotLight(LightSource):
    def __init__(self, intensity, position, direction, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.direction = normalize(direction)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from a point to the light source
    def get_light_ray(self, intersection):
        return Ray(intersection, normalize(self.position - intersection))


    def get_distance_from_light(self, intersection):
        return np.linalg.norm(intersection - self.position)


    def get_intensity(self, intersection):
        distance = self.get_distance_from_light(intersection)
        f_att = self.kc + self.kl*distance + self.kq * (distance**2)
        vector = normalize(self.get_light_ray(intersection).direction)
        return self.intensity * (np.dot(vector, -self.direction) / f_att)


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    # The function is getting the collection of objects in the scene and looks for the one with minimum distance.
    # The function should return the nearest object and its distance (in two different arguments)
    def nearest_intersected_object(self, objects):
        nearest_object = None
        min_distance = np.inf
        for obj in objects:
            intersection = obj.intersect(self)
            if intersection is not None:
                distance, hit_object = intersection
                if distance < min_distance:
                    nearest_object = hit_object
                    min_distance = distance
        return nearest_object, min_distance


class Object3D:
    def set_material(self, ambient, diffuse, specular, shininess, reflection):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection


class Plane(Object3D):
    def __init__(self, normal, point):
        self.normal = np.array(normal)
        self.point = np.array(point)

    def intersect(self, ray: Ray):
        v = self.point - ray.origin
        t = np.dot(v, self.normal) / (np.dot(self.normal, ray.direction) + 1e-6)
        if t > 0:
            return t, self
        else:
            return None

    def get_normal(self, hit_point):
        return self.normal

class Triangle(Object3D):
    """
        C
        /\
       /  \
    A /____\ B

    The fornt face of the triangle is A -> B -> C.
    
    """
    def __init__(self, a, b, c):
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)
        self.normal = self.compute_normal()

    # computes normal to the trainagle surface. Pay attention to its direction!
    def compute_normal(self):
        return normalize(np.cross(self.b - self.a, self.c - self.a))

    def intersect(self, ray: Ray):
        plane_formula = Plane(self.normal, self.a)
        intersection = plane_formula.intersect(ray)

        if intersection is None:
            return None
        
        else :
            p = ray.origin + ray.direction * intersection[0]
            areaABC = np.linalg.norm(np.cross((self.b - self.a), (self.c - self.a))) / 2
            alpha = np.linalg.norm(np.cross((p - self.b), (p - self.c))) / (2 * areaABC)
            beta = np.linalg.norm(np.cross((p - self.c), (p - self.a))) / (2 * areaABC)
            gamma = np.linalg.norm(np.cross((p - self.a), (p - self.b))) / (2 * areaABC)

            if alpha >= 0 and alpha <= 1 and beta >= 0 and beta <= 1 and gamma >= 0 and gamma <= 1 and np.abs(alpha + beta + gamma - 1) <= 1e-6 :
                return intersection[0], self
            else:
                return None

    def get_normal(self, hit_point):
        return self.normal

class Pyramid(Object3D):
    """     
            D
            /\*\
           /==\**\
         /======\***\
       /==========\***\
     /==============\****\
   /==================\*****\
A /&&&&&&&&&&&&&&&&&&&&\ B &&&/ C
   \==================/****/
     \==============/****/
       \==========/****/
         \======/***/
           \==/**/
            \/*/
             E 
    
    Similar to Traingle, every from face of the diamond's faces are:
        A -> B -> D
        B -> C -> D
        A -> C -> B
        E -> B -> A
        E -> C -> B
        C -> E -> A
    """
    def __init__(self, v_list):
        self.v_list = v_list
        self.triangle_list = self.create_triangle_list()

    def create_triangle_list(self):
        l = []
        t_idx = [
                [0,1,3],
                [1,2,3],
                [0,3,2],
                 [4,1,0],
                 [4,2,1],
                 [2,4,0]]
        
        for l_idx in t_idx:
            triangle = Triangle(self.v_list[l_idx[0]], self.v_list[l_idx[1]], self.v_list[l_idx[2]])
            l.append(triangle)
            
        return l

    def apply_materials_to_triangles(self):
        for triangle in self.triangle_list:
            triangle.set_material(self.ambient, self.diffuse, self.specular, self.shininess, self.reflection)

    def intersect(self, ray: Ray):
        min_intersection = None
        for triangle in self.triangle_list:
            intersection = triangle.intersect(ray)
            if intersection is None:
                continue

            t, _ = intersection
            if min_intersection == None or t < min_intersection[0]:
                min_intersection = intersection
                self.normal = triangle.compute_normal()
                
        return min_intersection
    
    def get_normal(self, hit_point):
        return self.normal
    
class Sphere(Object3D):
    def __init__(self, center, radius: float):
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray):
        center_to_origin = self.center - ray.origin
        projection_length = np.dot(center_to_origin, ray.direction)
        perpendicular_distance_squared = np.dot(center_to_origin, center_to_origin) - projection_length * projection_length
        if perpendicular_distance_squared > self.radius ** 2:
            return None
        half_chord_length = np.sqrt(self.radius ** 2 - perpendicular_distance_squared)
        first_intersection_point = projection_length - half_chord_length
        second_intersection_point2 = projection_length + half_chord_length
        if first_intersection_point < 0:
            intersection_point1 = second_intersection_point2
        if first_intersection_point < 0:
            return None
        return first_intersection_point, self
    
    def get_normal(self, hit_point):
        return hit_point - self.center

