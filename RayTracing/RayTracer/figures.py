import numpy as np
from gl import V3
from gl import WHITE
import mathLibraries as ml

OPAQUE = 0
REFLECTIVE = 1
TRANSPARENT = 2

WHITE = (1,1,1)

class DirectionalLight(object):
    def __init__(self, direction = V3(0,-1,0), intensity = 1, color = WHITE ):
        self.direction = direction / ml.norm(direction)
        self.intensity = intensity
        self.color = color

class AmbientLight(object):
    def __init__(self, strength = 0, color = WHITE):
        self.strength = strength
        self.color = color

    def getColor(self):
        return (self.strength * self.color[0],
                self.strength * self.color[1],
                self.strength * self.color[2])

class PointLight(object):
    # Luz con punto de origen que va en todas direcciones
    def __init__(self, position = V3(0,0,0), intensity = 1, color = WHITE):
        self.position = position
        self.intensity = intensity
        self.color = color

class Material(object):
     def __init__(self, diffuse = WHITE, spec = 1, ior = 1, matType = OPAQUE):
        self.diffuse = diffuse
        self.spec = spec
        self.ior = ior
        self.matType = matType


class Intersect(object):
    def __init__(self, distance, point, normal, sceneObject):
        self.distance = distance
        self.point = point
        self.normal = normal
        self.sceneObject = sceneObject

class Sphere(object):
    def __init__(self, center, radius, material = Material()):
        self.center = center
        self.radius = radius
        self.material = material

    def ray_intersect(self, orig, dir):

        L = ml.subVectors(self.center, orig)
        l = ml.norm(L)

        tca = ml.dotProduct(L, dir)

        d = (l**2 - tca**2)
        if d > self.radius ** 2:
            return None

        thc = (self.radius**2 - d) ** 0.5
        t0 = tca - thc
        t1 = tca + thc

        if t0 < 0:
            t0 = t1

        if t0 < 0:
            return None

        # P = O + t * D
        hit = ml.sumVectors(orig, t0 * np.array(dir) )
        normal = ml.subVectors( hit, self.center )
        normal = normal / ml.norm(normal) #la normalizo

        return Intersect( distance = t0,
                          point = hit,
                          normal = normal,
                          sceneObject = self)

class Plane(object):
    def __init__(self, position, normal, material = Material()):
        self.position = position
        self.normal = normal / ml.norm(normal)
        self.material = material

    def ray_intersect(self, orig, dir):
        #t = (( planePos - origRayo) dot planeNormal) / (dirRayo dot planeNormal)
        denom = ml.dotProduct(dir, self.normal)

        if abs(denom) > 0.0001:
            num = ml.dotProduct(ml.subVectors(self.position, orig), self.normal)
            t = num / denom
            if t > 0:
                # P = O + t * D
                hit = ml.sumVectors(orig, t * np.array(dir))

                return Intersect(distance = t,
                                 point = hit,
                                 normal = self.normal,
                                 sceneObject = self)

        return None



class AABB(object):
    # Axis Aligned Bounding Box
    def __init__(self, position, size, material = Material()):
        self.position = position
        self.size = size
        self.material = material
        self.planes = []

        self.boundsMin = [0,0,0]
        self.boundsMax = [0,0,0]

        halfSizeX = size[0] / 2
        halfSizeY = size[1] / 2
        halfSizeZ = size[2] / 2

        #Sides
        self.planes.append(Plane( ml.sumVectors(position, V3(halfSizeX,0,0)), V3(1,0,0), material))
        self.planes.append(Plane( ml.sumVectors(position, V3(-halfSizeX,0,0)), V3(-1,0,0), material))

        # Up and down
        self.planes.append(Plane( ml.sumVectors(position, V3(0,halfSizeY,0)), V3(0,1,0), material))
        self.planes.append(Plane( ml.sumVectors(position, V3(0,-halfSizeY,0)), V3(0,-1,0), material))

        # Front and Back
        self.planes.append(Plane( ml.sumVectors(position, V3(0,0,halfSizeZ)), V3(0,0,1), material))
        self.planes.append(Plane( ml.sumVectors(position, V3(0,0,-halfSizeZ)), V3(0,0,-1), material))

        #Bounds
        epsilon = 0.001
        for i in range(3):
            self.boundsMin[i] = self.position[i] - (epsilon + self.size[i]/2)
            self.boundsMax[i] = self.position[i] + (epsilon + self.size[i]/2)


    def ray_intersect(self, orig, dir):
        intersect = None
        t = float('inf')

        for plane in self.planes:
            planeInter = plane.ray_intersect(orig, dir)
            if planeInter is not None:
                # Si estoy dentro de los bounds
                if planeInter.point[0] >= self.boundsMin[0] and planeInter.point[0] <= self.boundsMax[0]:
                    if planeInter.point[1] >= self.boundsMin[1] and planeInter.point[1] <= self.boundsMax[1]:
                        if planeInter.point[2] >= self.boundsMin[2] and planeInter.point[2] <= self.boundsMax[2]:
                            #Si soy el plano mas cercano
                            if planeInter.distance < t:
                                t = planeInter.distance
                                intersect = planeInter


        if intersect is None:
            return None

        return Intersect(distance = intersect.distance,
                         point = intersect.point,
                         normal = intersect.normal,
                         sceneObject = self)