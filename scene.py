from math import *
from manim import *

#manim -pqk scene.py 
#manim -pqk --format=png scene.py 

# Colors
WHITE = '#FFFFFF'
BLACK = '#000000'
RED = '#C02020'
GREEN = '#70FF70'
BLUE = '#8080FF'
PURE_RED = '#FF0000'
PURE_GREEN = '#00FF00'
PURE_BLUE = '#0000FF'

class Cover(Scene):
    def construct(self):
        self.camera.background_color = '#101020'
        p0 = Circle(radius=0.25, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=10).move_to([-6,-1,0]).set_z_index(2)
        p1 = Circle(radius=0.25, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=10).move_to([0,-3,0]).set_z_index(2)
        p2 = Circle(radius=0.25, color=BLACK, fill_opacity=1, stroke_color=BLUE, stroke_width=10).move_to([0,3,0]).set_z_index(2)
        p3 = Circle(radius=0.25, color=BLACK, fill_opacity=1, stroke_color=BLUE, stroke_width=10).move_to([6,1,0]).set_z_index(2)
        self.add(p0,p1,p2,p3)
        l0 = Line(p0.get_center(),p1.get_center(),stroke_width=10)
        l1 = Line(p2.get_center(),p3.get_center(),stroke_width=10)
        self.add(l0,l1)
        self.add(CubicBezier([-6,-1,0],[0,-3,0],[0,3,0],[6,1,0], stroke_width=15, fill_color=WHITE).set_z_index(1))
        self.add(Arrow([0,0,0],np.array([0.832,0.555,0])*5,buff=0,color=GREEN,stroke_width=15))


class Slide2_2(Scene):
    def construct(self):
        p0 = Circle(radius=0.25, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=10).move_to([-6,-1,0]).set_z_index(2)
        p1 = Circle(radius=0.25, color=BLACK, fill_opacity=1, stroke_color=BLUE, stroke_width=10).move_to([6,1,0]).set_z_index(2)
        self.add(p0,p1)
        self.play(Create(CubicBezier([-6,-1,0],[0,-4,0],[0,4,0],[6,1,0], stroke_width=15, fill_color=WHITE).set_z_index(1)), run_time=2)

class Slide3_1(Scene):
    def construct(self):
        p0 = Circle(radius=0.25, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=10).move_to([-5,-2,0]).set_z_index(2)
        p1 = Circle(radius=0.25, color=BLACK, fill_opacity=1, stroke_color=BLUE, stroke_width=10).move_to([5,2,0]).set_z_index(2)
        l0 = MathTex("A", color=RED).move_to([-5,-1.3,0])
        l1 = MathTex("B", color=BLUE).move_to([5,2.7,0])
        self.add(p0,p1,l0,l1)

class Slide3_4(Scene):
    def construct(self):
        ax = Axes(
            x_range=[0, 1, 0.2],
            y_range=[0, 1, 0.2],
            x_length=6,
            y_length=6,
            tips=False,
            axis_config={"include_numbers": True},
        ).set_z_index(2)
        xLabel = MathTex("t").move_to([3.5,-3,0])
        yLabel = MathTex("w").move_to([-3,3.5,0])

        self.add(ax, xLabel, yLabel)

        self.play(
            Create(Line(
                ax.coords_to_point(0,1),
                ax.coords_to_point(1,0),
                color=RED
            )),
            Create(Line(
                ax.coords_to_point(0,0),
                ax.coords_to_point(1,1),
                color=BLUE
            )),
            run_time=2
        )
