import struct
from collections import namedtuple

from numpy.core.defchararray import add

from obj import Obj

import numpy as np

from numpy import sin, cos, tan

import mathLibraries as ml

import random

STEPS = 1

OPAQUE = 0
REFLECTIVE = 1
TRANSPARENT = 2

MAX_RECURSION_DEPTH = 3

pi = 3.1415926535897932384626433

V2 = namedtuple('Point2', ['x', 'y'])
V3 = namedtuple('Point3', ['x', 'y', 'z'])
V4 = namedtuple('Point4', ['x', 'y', 'z', 'w'])

def char(c):
    # 1 byte
    return struct.pack('=c', c.encode('ascii'))

def word(w):
    #2 bytes
    return struct.pack('=h', w)

def dword(d):
    # 4 bytes
    return struct.pack('=l', d)

def _color(r, g, b):
    # Acepta valores de 0 a 1
    # Se asegura que la información de color se guarda solamente en 3 bytes
    return bytes([ int(b * 255), int(g* 255), int(r* 255)])

def baryCoords(A, B, C, P):
    # u es para A, v es para B, w es para C
    try:
        #PCB/ABC
        u = (((B.y - C.y) * (P.x - C.x) + (C.x - B.x) * (P.y - C.y)) /
            ((B.y - C.y) * (A.x - C.x) + (C.x - B.x) * (A.y - C.y)))

        #PCA/ABC
        v = (((C.y - A.y) * (P.x - C.x) + (A.x - C.x) * (P.y - C.y)) /
            ((B.y - C.y) * (A.x - C.x) + (C.x - B.x) * (A.y - C.y)))

        w = 1 - u - v
    except:
        return -1, -1, -1

    return u, v, w

def reflectVector(normal, dirVector):
    # R = 2 * ( N . L) * N - L
    reflect1 = 2 * ml.dotProduct(normal, dirVector)
    reflect2 = ml.kMul(normal, reflect1)
    reflect = ml.subVectors(reflect2, dirVector)
    #reflect = reflect / np.linalg.norm(reflect)
    reflect = ml.norm(reflect)
    return reflect


def refractVector(normal, dirVector, ior):
    # Snell's Law
    cosi = max(-1, min(1 , ml.dotProduct(dirVector, normal)))
    etai = 1
    etat = ior

    if cosi < 0:
        cosi = -cosi
    else:
        etai, etat = etat, etai
        #normal = np.array(normal) * -1
        normal = ml.kMul(normal,-1)

    eta = etai/etat
    k = 1 - eta * eta * (1 - (cosi * cosi))

    if k < 0: #Total Internal Reflection
        return None

    #R = eta * np.array(dirVector) + (eta * cosi - k**0.5) * normal
    multiplication = ml.kMul(dirVector, eta)
    multiplication1 = ml.kMul(normal, (eta * cosi - k**0.5))
    addition = ml.sumVectors(multiplication, multiplication1)
    R = addition
    R = ml.norm(R)

    return R #R / np.linalg.norm(R)


def fresnel(normal, dirVector, ior):
    cosi = max(-1, min(1 , ml.dotProduct(dirVector, normal)))
    etai = 1
    etat = ior

    if cosi > 0:
        etai, etat = etat, etai

    sint = etai / etat * (max(0, 1 - cosi * cosi) ** 0.5)

    if sint >= 1: #Total internal reflection
        return 1

    cost = max(0, 1 - sint * sint) ** 0.5
    cosi = abs(cosi)
    Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost))
    Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost))

    return (Rs * Rs + Rp * Rp) / 2


BLACK = _color(0,0,0)
WHITE = _color(1,1,1)


class Raytracer(object):
    def __init__(self, width, height):
        #Constructor
        self.curr_color = WHITE
        self.clear_color = BLACK
        self.glCreateWindow(width, height)

        self.camPosition = V3(0,0,0)
        self.fov = 60

        self.background = None

        self.scene = []

        self.pointLights = []
        self.ambLight = None
        self.dirLight = None

        self.envmap = None


    def glFinish(self, filename):
        #Crea un archivo BMP y lo llena con la información dentro de self.pixels
        with open(filename, "wb") as file:
            # Header
            file.write(bytes('B'.encode('ascii')))
            file.write(bytes('M'.encode('ascii')))
            file.write(dword(14 + 40 + (self.width * self.height * 3)))
            file.write(dword(0))
            file.write(dword(14 + 40))

            # InfoHeader
            file.write(dword(40))
            file.write(dword(self.width))
            file.write(dword(self.height))
            file.write(word(1))
            file.write(word(24))
            file.write(dword(0))
            file.write(dword(self.width * self.height * 3))
            file.write(dword(0))
            file.write(dword(0))
            file.write(dword(0))
            file.write(dword(0))

            # Color Table
            for y in range(self.height):
                for x in range(self.width):
                    file.write( _color(self.pixels[x][y][0],
                                       self.pixels[x][y][1],
                                       self.pixels[x][y][2]))


    def glCreateWindow(self, width, height):
        self.width = width
        self.height = height
        self.glClear()
        self.glViewport(0,0, width, height)


    def glViewport(self, x, y, width, height):
        self.vpX = int(x)
        self.vpY = int(y)
        self.vpWidth = int(width)
        self.vpHeight = int(height)


    def glClearColor(self, r, g, b):
        self.clear_color = _color(r, g, b)


    def glClear(self):
        #Crea una lista 2D de pixeles y a cada valor le asigna 3 bytes de color
        self.pixels = [[ self.clear_color for y in range(self.height)]
                         for x in range(self.width)]

    def glClearBackground(self):
        if self.background:
            for x in range(self.vpX, self.vpX + self.vpWidth):
                for y in range(self.vpY, self.vpY + self.vpHeight):

                    tx = (x - self.vpX) / self.vpWidth
                    ty = (y - self.vpY) / self.vpHeight

                    self.glPoint(x,y, self.background.getColor(tx, ty))



    def glViewportClear(self, color = None):
        for x in range(self.vpX, self.vpX + self.vpWidth):
            for y in range(self.vpY, self.vpY + self.vpHeight):
                self.glPoint(x,y, color)


    def glColor(self, r, g, b):
        self.curr_color = _color(r,g,b)

    def glPoint(self, x, y, color = None):
        if x < self.vpX or x >= self.vpX + self.vpWidth or y < self.vpY or y >= self.vpY + self.vpHeight:
            return

        if (0 <= x < self.width) and (0 <= y < self.height):
            self.pixels[int(x)][int(y)] = color or self.curr_color


    def glRender(self):

        for y in range(0, self.height, STEPS):
            for x in range(0, self.width , STEPS):
                # pasar de coordenadas de ventana a coordenadas NDC (-1 a 1)
                Px = 2 * ((x + 0.5) / self.width) - 1
                Py = 2 * ((y + 0.5) / self.height) - 1

                # Angulo de vision, asumiendo que el near plane esta a 1 unidad de la camara
                t = tan( (self.fov *  pi / 180) / 2)
                r = t * self.width / self.height

                Px *= r
                Py *= t

                #La camara siempre esta viendo hacia -Z
                direction = V3(Px, Py, -1)
                #direction = direction / np.linalg.norm(direction)
                direction = ml.norm(direction)

                self.glPoint(x,y, self.cast_ray(self.camPosition, direction))

    def scene_intersect(self, orig, dir, origObj = None):
        depth = float('inf')
        intersect = None

        for obj in self.scene:
            if obj is not origObj:
                hit = obj.ray_intersect(orig,dir)
                if hit != None:
                    if hit.distance < depth:
                        depth = hit.distance
                        intersect = hit

        return intersect


    def cast_ray(self, orig, dir, origObj = None, recursion = 0):
        intersect = self.scene_intersect(orig, dir, origObj)

        if intersect == None or recursion >= MAX_RECURSION_DEPTH:
            if self.envmap:
                return self.envmap.getColor(dir)
            return self.clear_color

        material = intersect.sceneObject.material

        # Colors
        #finalColor = np.array([0,0,0])
        finalColor = V3(0,0,0)
        #objectColor = np.array([material.diffuse[0],
                                #material.diffuse[1],
                                #material.diffuse[2]])
        objectColor = V3(material.diffuse[0], material.diffuse[1], material.diffuse[2])
        

        #ambientColor = np.array([0,0,0])
        ambientColor = V3(0,0,0)
        #dirLightColor = np.array([0,0,0])
        dirLightColor = V3(0,0,0)
        #pLightColor = np.array([0,0,0])
        pLightColor = V3(0,0,0)
        #finalSpecColor = np.array([0,0,0])
        finalSpecColor = V3(0,0,0)
        #reflectColor = np.array([0,0,0])
        reflectColor = V3(0,0,0)
        #refractColor = np.array([0,0,0])
        refractColor = V3(0,0,0)

        # Direccion de vista
        view_dir = ml.subVectors(self.camPosition, intersect.point)
        #view_dir = view_dir / np.linalg.norm(view_dir)
        view_dir = ml.norm(view_dir)

        if self.ambLight:
            #ambientColor = np.array(self.ambLight.getColor())
            ambientColor = self.ambLight.getColor()

        if self.dirLight:
            #diffuseColor = np.array([0,0,0])
            diffuseColor = V3(0,0,0)
            #specColor = np.array([0,0,0])
            specColor = V3(0,0,0)
            shadow_intensity = 0

            # Iluminacion difusa
            #light_dir = np.array( self.dirLight.direction) * -1
            #minusOne= [-1,-1,-1]
            light_dir = ml.kMul(self.dirLight.direction, -1)
            intensity = max(0, ml.dotProduct(intersect.normal, light_dir)) * self.dirLight.intensity
            #diffuseColor = np.array([intensity * self.dirLight.color[0],
                                     #intensity * self.dirLight.color[1],
                                     #intensity * self.dirLight.color[2]])
            diffuseColor = V3(intensity * self.dirLight.color[0],
                              intensity * self.dirLight.color[1],
                              intensity * self.dirLight.color[2])


            # Iluminacion especular
            reflect = reflectVector(intersect.normal, light_dir)
            spec_intensity = self.dirLight.intensity * max(0,ml.dotProduct(view_dir, reflect)) ** material.spec
            #specColor = np.array([spec_intensity * self.dirLight.color[0],
             #                     spec_intensity * self.dirLight.color[1],
              #                    spec_intensity * self.dirLight.color[2]])
            specColor = V3(spec_intensity * self.dirLight.color[0],
                           spec_intensity * self.dirLight.color[1],
                           spec_intensity * self.dirLight.color[2])


            # Shadow
            shadInter = self.scene_intersect(intersect.point, light_dir, intersect.sceneObject)
            if shadInter:
                shadow_intensity = 1

            #dirLightColor = (1 - shadow_intensity) * diffuseColor
            shadow_intensity = 1 - shadow_intensity 
            dirLightColor = ml.kMul(diffuseColor, shadow_intensity)
            finalSpecColor = ml.sumVectors(finalSpecColor, ml.kMul(specColor, shadow_intensity))

        for pointLight in self.pointLights:
            #diffuseColor = np.array([0,0,0])
            diffuseColor = V3(0,0,0)
            #specColor = np.array([0,0,0])
            specColor = V3(0,0,0)
            shadow_intensity = 0

            # Iluminacion difusa
            light_dir = ml.subVectors(pointLight.position, intersect.point)
            #light_dir = light_dir / np.linalg.norm(light_dir)
            light_dir = ml.norm(light_dir)
            intensity = max(0, ml.dotProduct(intersect.normal, light_dir)) * pointLight.intensity
            #diffuseColor = np.array([intensity * pointLight.color[0],
             #                        intensity * pointLight.color[1],
              #                       intensity * pointLight.color[2]])
            diffuseColor = V3(intensity * pointLight.color[0],
                              intensity * pointLight.color[1],
                              intensity * pointLight.color[2])


            # Iluminacion especular
            reflect = reflectVector(intersect.normal, light_dir)
            spec_intensity = pointLight.intensity * max(0,ml.dotProduct(view_dir, reflect)) ** material.spec
            #specColor = np.array([spec_intensity * pointLight.color[0],
             #                     spec_intensity * pointLight.color[1],
             #                     spec_intensity * pointLight.color[2]])
            specColor = V3(spec_intensity * pointLight.color[0],
                           spec_intensity * pointLight.color[1],
                           spec_intensity * pointLight.color[2])


            # Shadows
            shadInter = self.scene_intersect(intersect.point, light_dir, intersect.sceneObject)
            lightDistance = ml.length( ml.subVectors(pointLight.position, intersect.point) )
            if shadInter and shadInter.distance < lightDistance:
                shadow_intensity = 1

            shadow_intensity = 1 - shadow_intensity 
            pLightColor = ml.sumVectors(pLightColor,  ml.kMul(diffuseColor, shadow_intensity))
            finalSpecColor = ml.sumVectors(finalSpecColor, ml.kMul(specColor, shadow_intensity))


        if material.matType == OPAQUE:
            finalColor1 = ml.sumVectors(pLightColor, ambientColor )
            finalColor2 =  ml.sumVectors(dirLightColor, finalColor1)
            finalColor = ml.sumVectors(finalColor2, finalSpecColor)


        elif material.matType == REFLECTIVE:
            #reflect = reflectVector(intersect.normal, np.array(dir) * -1)
            dirMinusOne = ml.kMul(dir, -1)
            reflect = reflectVector(intersect.normal, dirMinusOne)
            reflectColor = self.cast_ray(intersect.point, reflect, intersect.sceneObject, recursion + 1)
            #reflectColor = np.array([reflectColor[0],
             #                        reflectColor[1],
              #                       reflectColor[2]])
            reflectColor = V3(reflectColor[0], reflectColor[1], reflectColor[2])

            finalColor = ml.sumVectors(reflectColor,  finalSpecColor)

        elif material.matType == TRANSPARENT:
            outside = ml.dotProduct(dir, intersect.normal) < 0
            #bias = 0.001 * intersect.normal
            bias = ml.kMul(intersect.normal, 0.001)
            kr = fresnel(intersect.normal, dir, material.ior)

            dirMinusOne = ml.kMul(dir, -1)
            reflect = reflectVector(intersect.normal, dirMinusOne)
            #reflect = reflectVector(intersect.normal, np.array(dir) * -1)
            reflectOrig = ml.sumVectors(intersect.point, bias) if outside else ml.subVectors(intersect.point, bias)
            reflectColor = self.cast_ray(reflectOrig, reflect, None, recursion + 1)
            #reflectColor = np.array(reflectColor)
            reflectColor = V3(reflectColor[0], reflectColor[1], reflectColor[2])

            if kr < 1:
                refract = refractVector(intersect.normal, dir, material.ior )
                refractOrig = ml.subVectors(intersect.point, bias) if outside else ml.sumVectors(intersect.point, bias)
                refractColor = self.cast_ray(refractOrig, refract, None, recursion + 1)
                #reflectColor = np.linalg.norm(reflectColor)
                refractColor = V3(refractColor[0], refractColor[1], refractColor[2])

            #finalColor = reflectColor * kr + refractColor * (1 - kr) + finalSpecColor
            finalColor = ml.sumVectors(ml.kMul(reflectColor, kr), ml.kMul(refractColor, (1-kr)))
            finalColor = ml.sumVectors(finalColor, finalSpecColor)




        # Le aplicamos el color del objeto
        #finalColor *= objectColor
        finalColor = ml.vectorMultiplication(objectColor, finalColor)

        #Nos aseguramos que no suba el valor de color de 1
        r = min(1, finalColor[0])
        g = min(1, finalColor[1])
        b = min(1, finalColor[2])
    
        return (r,g,b)































