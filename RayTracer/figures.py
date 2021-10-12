import struct
from collections import namedtuple
import mathLibraries as ml
import numpy as np


V2 = namedtuple('Point2', ['x', 'y'])
V3 = namedtuple('Point3', ['x', 'y', 'z'])
V4 = namedtuple('Point4', ['x', 'y', 'z', 'w'])


OPAQUE = 0  
REFLECTIVE = 1
TRANSPARENT = 2

WHITE = (1,1,1)

pi = 3.1415926535897932384626433


class DirectionalLight(object):
    def __init__(self, direction=V3(0,-1,0), intensity = 1,color = WHITE):
        self.direction = ml.norm(direction)
        self.intensity = intensity
        self.color = color

class AmbientLight(object):
    def __init__(self, strength = 0, color = WHITE):
        self.strength = strength
        self.color = color
    
    def getColor(self):
        return (V3(self.strength * self.color[0],
                   self.strength * self.color[1],
                   self.strength * self.color[2]))

class PointLight(object):
    # Luz con putno de origen que va en todas direcciones
    def __init__(self, position = V3(0,0,0), intensity = 1, color = WHITE):
        self.position = position
        self.intensity = intensity
        self.color = color

class Material(object):
    def __init__(self, diffuse = WHITE, spec = 1, ior = 1, texture = None, matType = OPAQUE):
        self.diffuse = diffuse
        self.spec = spec
        self.ior = ior
        self.texture = texture
        self.matType = matType

class Intersect(object):
    def __init__(self, distance, point, normal, texCoords, sceneObject):
        self.distance = distance
        self.point = point
        self.normal = normal
        self.texCoords = texCoords
        self.sceneObject = sceneObject

class Sphere(object):
    def __init__(self, center, radius, material = Material()):
        self.center = center
        self.radius = radius
        self.material = material

    def ray_intersect(self, orig, dir):
        
        L = ml.subVectors(self.center, orig)
        l = ml.length(L)

        tca = ml.dotProduct(L, dir)

        d = (l**2 - tca**2) 

        if d > self.radius ** 2:
            return None

        thc = (self.radius**2 - d)**0.5
        t0 = tca - thc
        t1 = tca + thc

        #La esfera esta completamente detras de la camara
        
        if t0 < 0:
            t0 = t1

        if t0 < 0:
            return None

        # La camara esta dentro de la esfera 
        # P = O + t * D
        hit = ml.sumVectors(orig, ml.kMul(dir, t0))
        normal = ml.norm(ml.subVectors(hit, self.center))

        u = 1 - ((np.arctan2(normal[2], normal[0] ) / (2 * pi)) + 0.5)
        v = np.arccos(-normal[1]) / pi

        uvs = (u,v)

        return Intersect( distance = t0,
                          point = hit,
                          normal = ml.norm(normal),
                          texCoords = uvs,
                          sceneObject = self)

#class Triangle(object):
#    def __init__(self, position, size, material = Material()):
#        self.position = position
#        self.material = material
#        
#    def ray_intersect(self, orig, dir, v0, v1, v2):
#        # t is the distance from the ray origin O to P
#        # P is somewhere between on the ray/ this is also the intersection point of the ray
#        # O is the origin of the ray
#        # dir is the direction of the ray
#        # D is the distance from origin to the plane
#        # N is the normal of the plane
#
#
#        v0v1 = ml.subVectors(self.v1, self.v0)
#        v0v2 = ml.subVectors(self.v2, self.v0)
#
#        N = ml.crossProduct(v0v1, v0v2)
#        area2 = N.length()
#
#        NRay = ml.dotProduct(N, dir)
#
#        if (abs(NRay < 0.09)):
#            return False
#
#        d = ml.dotProduct(N, self.v0)
#
#        t = ml.dotProduct(N, orig)
#        t = ml.sumVectors(t, d)
#
#        if t < 0:
#            return False
#        
#        P = ml.sumVectors(orig, t)
#        P = ml.vectorMultiplication(P, dir)
#
#        # Edge 0
#        edge0 =  ml.subVectors(self.v1, self.v0)
#        vp0 = ml.subVectors(P, self.v1)
#        C = ml.crossProduct(edge0, vp0)
#        
#        if (ml.dotProduct(N, C) < 0):
#            return False
#
#        # Edge 1
#        edge1 = ml.subVectors(self.v2, self.v1)
#        vp1 = ml.subVectors(P, self.v2)
#        C = ml.crossProduct(edge1, vp1)
#        
#        if (ml.dotProduct(N, C) < 0):
#            return False
#
#        # Edge 2
#        edge2 = ml.subVectors(self.v0, self.v2)
#        vp2 = ml.subVectors(P, self.v2)
#        C = ml.crossProduct(edge2, vp2)
#
#        if (ml.dotProduct(N, C) < 0):
#            return False
#
#        return Intersect(distance = t,
#                        point = P,
#                        normal = self.normal,
#                        #texCoords = None,
#                        sceneObject = self)

class Plane(object):
    def __init__(self, postition, normal, material = Material()):
        self.position = postition
        self.normal = ml.norm(normal)
        self.material = material
    
    def ray_intersect(self, orig, dir):
        #t = ((planPos - origRayo) dot planeNOrmal)) / (dirRayo dot planeNormal) 
        denom = ml.dotProduct(dir, self.normal)

        if abs(denom) > 0.0001:
            num = ml.dotProduct(ml.subVectors(self.position, orig), self.normal)
            t = num / denom
            if t > 0:
                #P = O + t * D
                hit = ml.sumVectors(orig, ml.kMul(dir, t))

                return Intersect(distance = t,
                                 point = hit,
                                 normal = self.normal,
                                 texCoords = None,
                                 sceneObject = self)
        return None

class AABB(object):
    #Axis Aligned Bounding Bos
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
        self.planes.append(Plane(ml.sumVectors(position,V3(halfSizeX,0,0)), V3(1,0,0), material))
        self.planes.append(Plane(ml.sumVectors(position,V3(-halfSizeX,0,0)), V3(-1,0,0), material))

        #Up and Down
        self.planes.append(Plane(ml.sumVectors(position,V3(0,halfSizeY,0)), V3(0,1,0), material))
        self.planes.append(Plane(ml.sumVectors(position,V3(0,-halfSizeY,0)), V3(0,-1,0), material))

        #Front and Back
        self.planes.append(Plane(ml.sumVectors(position,V3(0,0,halfSizeZ)), V3(0,0,1), material))
        self.planes.append(Plane(ml.sumVectors(position,V3(0,0,-halfSizeZ)), V3(0,0,-1), material))

        #Bounds
        epsilon = 0.001
        for i in range(3):
            self.boundsMin[i] = self.position[i] - (epsilon + self.size[i]/2)
            self.boundsMax[i] = self.position[i] + (epsilon + self.size[i]/2)

    def ray_intersect(self, orig, dir):
        intersect = None
        t = float('inf')

        uvs = None

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

                                u, v = 0, 0

                                if abs(plane.normal[0]) > 0:
                                    # mapear uvs para eje X, uso coordenadas en Y y Z.
                                    u = (planeInter.point[1] - self.boundsMin[1]) / (self.boundsMax[1] - self.boundsMin[1])
                                    v = (planeInter.point[2] - self.boundsMin[2]) / (self.boundsMax[2] - self.boundsMin[2])

                                elif abs(plane.normal[1]) > 0:
                                    # mapear uvs para eje Y, uso coordenadas en X y Z.
                                    u = (planeInter.point[0] - self.boundsMin[0]) / (self.boundsMax[0] - self.boundsMin[0])
                                    v = (planeInter.point[2] - self.boundsMin[2]) / (self.boundsMax[2] - self.boundsMin[2])

                                elif abs(plane.normal[2]) > 0:
                                    # mapear uvs para eje Z, uso coordenadas en X y Y.
                                    u = (planeInter.point[0] - self.boundsMin[0]) / (self.boundsMax[0] - self.boundsMin[0])
                                    v = (planeInter.point[1] - self.boundsMin[1]) / (self.boundsMax[1] - self.boundsMin[1])

                                uvs = (u,v)

        if intersect is None: 
            return None
        
        return Intersect(distance = intersect.distance,
                         point = intersect.point,
                         normal = intersect.normal,
                         texCoords = uvs,
                         sceneObject = self)