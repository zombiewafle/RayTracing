from gl import Raytracer, V3, _color
from obj import Obj, Texture

from figures import Sphere, Material

width = 512
height = 512

#Ojos
sclera1 = Material(diffuse = _color(1,1,1))
sclera2 = Material(diffuse = _color(1,1,1))
pupil1 = Material(diffuse = _color(0,0,0))
pupil2 = Material(diffuse = _color(0,0,0))

#Cuerpo
body1 = Material(diffuse = _color(0.90,0.90,0.90))
body2 = Material(diffuse = _color(0.90,0.90,0.90))
body3 = Material(diffuse = _color(0.90,0.90,0.90))

#Nariz
nose = Material(diffuse = _color(1,0.6,0))

#Boca
smile1 = Material(diffuse = _color(0.5,0.5,0.5))
smile2 = Material(diffuse = _color(0.5,0.5,0.5))
smile3 = Material(diffuse = _color(0.5,0.5,0.5))
smile4 = Material(diffuse = _color(0.5,0.5,0.5))

#Botones
blackDot1 = Material(diffuse = _color(0,0,0))
blackDot2 = Material(diffuse = _color(0,0,0))
blackDot3 = Material(diffuse = _color(0,0,0))



rtx = Raytracer(width,height)

#Ojos
rtx.scene.append( Sphere(V3(-0.05, 0.25,-2), 0.025, sclera1) )
rtx.scene.append( Sphere(V3(0.05, 0.25,-2), 0.025, sclera2) )
rtx.scene.append( Sphere(V3(-0.03, 0.13,-1), 0.008, pupil1) )
rtx.scene.append( Sphere(V3(0.020, 0.13,-1), 0.008, pupil2) )

#Cuerpo
rtx.scene.append( Sphere(V3(0, 0.30, -3), 0.2, body1) )
rtx.scene.append( Sphere(V3(0, 0, -3), 0.25, body2) )
rtx.scene.append( Sphere(V3(0, -0.40, -3), 0.3, body3) )
#
##Nariz
rtx.scene.append( Sphere(V3(0, 0.20, -2), 0.03, nose) )
#
##Boca
rtx.scene.append( Sphere(V3(-0.03, 0.09, -1), 0.006, smile1) )
rtx.scene.append( Sphere(V3(-0.01, 0.08, -1), 0.006, smile2) )
rtx.scene.append( Sphere(V3(0.01, 0.08, -1), 0.006, smile3) )
rtx.scene.append( Sphere(V3(0.03, 0.09, -1), 0.006, smile4) )
#
##Botones
rtx.scene.append( Sphere(V3(0, 0.07 , -2), 0.03, blackDot1) )
rtx.scene.append( Sphere(V3(0, -0.055 , -2), 0.03, blackDot2) )
rtx.scene.append( Sphere(V3(0, -0.2 , -2), 0.03, blackDot3) )




rtx.glRender()

rtx.glFinish('output.bmp')



