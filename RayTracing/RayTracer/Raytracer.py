from gl import Raytracer, V3, _color
from obj import *
from figures import *


#width = 512
#height = 512
#
##Ojos
#sclera1 = Material(diffuse = _color(1,1,1))
#sclera2 = Material(diffuse = _color(1,1,1))
#pupil1 = Material(diffuse = _color(0,0,0))
#pupil2 = Material(diffuse = _color(0,0,0))
#
##Cuerpo
#body1 = Material(diffuse = _color(0.90,0.90,0.90))
#body2 = Material(diffuse = _color(0.90,0.90,0.90))
#body3 = Material(diffuse = _color(0.90,0.90,0.90))
#
##Nariz
#nose = Material(diffuse = _color(1,0.6,0))
#
##Boca
#smile1 = Material(diffuse = _color(0.5,0.5,0.5))
#smile2 = Material(diffuse = _color(0.5,0.5,0.5))
#smile3 = Material(diffuse = _color(0.5,0.5,0.5))
#smile4 = Material(diffuse = _color(0.5,0.5,0.5))
#
##Botones
#blackDot1 = Material(diffuse = _color(0,0,0))
#blackDot2 = Material(diffuse = _color(0,0,0))
#blackDot3 = Material(diffuse = _color(0,0,0))
#
#
#
#rtx = Raytracer(width,height)
#
##Ojos
#rtx.scene.append( Sphere(V3(-0.05, 0.25,-2), 0.025, sclera1) )
#rtx.scene.append( Sphere(V3(0.05, 0.25,-2), 0.025, sclera2) )
#rtx.scene.append( Sphere(V3(-0.03, 0.13,-1), 0.008, pupil1) )
#rtx.scene.append( Sphere(V3(0.020, 0.13,-1), 0.008, pupil2) )
#
##Cuerpo
#rtx.scene.append( Sphere(V3(0, 0.30, -3), 0.2, body1) )
#rtx.scene.append( Sphere(V3(0, 0, -3), 0.25, body2) )
#rtx.scene.append( Sphere(V3(0, -0.40, -3), 0.3, body3) )
##
###Nariz
#rtx.scene.append( Sphere(V3(0, 0.20, -2), 0.03, nose) )
##
###Boca
#rtx.scene.append( Sphere(V3(-0.03, 0.09, -1), 0.006, smile1) )
#rtx.scene.append( Sphere(V3(-0.01, 0.08, -1), 0.006, smile2) )
#rtx.scene.append( Sphere(V3(0.01, 0.08, -1), 0.006, smile3) )
#rtx.scene.append( Sphere(V3(0.03, 0.09, -1), 0.006, smile4) )
##
###Botones
#rtx.scene.append( Sphere(V3(0, 0.07 , -2), 0.03, blackDot1) )
#rtx.scene.append( Sphere(V3(0, -0.055 , -2), 0.03, blackDot2) )
#rtx.scene.append( Sphere(V3(0, -0.2 , -2), 0.03, blackDot3) )
#
#
#
#
#rtx.glRender()
#
#rtx.glFinish('output.bmp')
#
#
#
#


# Dimensiones
width = 512
height = 512

# Materiales
wood = Material(diffuse = (0.6,0.2,0.2), spec = 64)
stone = Material(diffuse = (0.4,0.4,0.4), spec = 64)

gold = Material(diffuse = (1, 0.8, 0 ),spec = 32, matType = REFLECTIVE)
mirror = Material(spec = 128, matType = REFLECTIVE)

water = Material(spec = 64, ior = 1.33, matType = TRANSPARENT)
glass = Material(spec = 64, ior = 1.5, matType = TRANSPARENT)
diamond = Material(spec = 64, ior = 2.417, matType = TRANSPARENT)


# Inicializacion
rtx = Raytracer(width,height)
rtx.envmap = EnvMap('envmap_playa.bmp')

# Luces
rtx.ambLight = AmbientLight(strength = 0.1)
rtx.dirLight = DirectionalLight(direction = V3(1, -1, -2), intensity = 0.5)
rtx.pointLights.append( PointLight(position = V3(0, 2, 0), intensity = 0.5))

# Objetos
rtx.scene.append( Sphere(V3(-2,2,-8), 1, mirror) )
rtx.scene.append( Sphere(V3(2,2,-8), 1, glass) )

rtx.scene.append( AABB(V3(-2,-2,-8), V3(2,2,2), mirror) )
rtx.scene.append( AABB(V3(2,-2,-8), V3(2,2,2), glass) )





# Terminar
rtx.glRender()
rtx.glFinish('output.bmp')


