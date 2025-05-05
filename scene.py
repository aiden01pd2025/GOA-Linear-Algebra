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
YELLOW = '#F0C000'
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

class Bill(ThreeDScene):
    def construct(self):
        #self.camera.light_source.move_to(OUT*30)
        self.set_camera_orientation(phi=PI*1/5, theta=PI*0)
        self.set_camera_orientation(zoom=0.25)

        s = 4
        h=9
        a = Variable(var=0.00, label='', num_decimal_places=2).move_to(h*10)
        self.add(a)

        panel1 = always_redraw(lambda : Surface(
            lambda u,v : np.array([
                s-s*u/2 ,
                s*sqrt(3)*u/2 ,
                max(0 , h*v-a.value.get_value()*(1-u))
            ]),
            u_range = [0,1],
            v_range = [0,1],
            checkerboard_colors=[ManimColor('#EBA851'), ManimColor('#F68002')]
        ))
        panel2 = always_redraw(lambda : Surface(
            lambda u,v : rotate_vector(np.array([
                s-s*u/2 ,
                s*sqrt(3)*u/2 ,
                max(0 , h*v-a.value.get_value()*u)
            ]), PI/3),
            u_range = [0,1],
            v_range = [0,1],
            checkerboard_colors=[ManimColor('#EBA851'), ManimColor('#F68002')]
        ))
        panel3 = always_redraw(lambda : Surface(
            lambda u,v : rotate_vector(np.array([
                s-s*u/2 ,
                s*sqrt(3)*u/2 ,
                max(0 , h*v-a.value.get_value()*(1-u))
            ]), PI*2/3),
            u_range = [0,1],
            v_range = [0,1],
            checkerboard_colors=[ManimColor('#EBA851'), ManimColor('#F68002')]
        ))
        panel4 = always_redraw(lambda : Surface(
            lambda u,v : rotate_vector(np.array([
                s-s*u/2 ,
                s*sqrt(3)*u/2 ,
                max(0 , h*v-a.value.get_value()*u)
            ]), PI*3/3),
            u_range = [0,1],
            v_range = [0,1],
            checkerboard_colors=[ManimColor('#EBA851'), ManimColor('#F68002')]
        ))
        panel5 = always_redraw(lambda : Surface(
            lambda u,v : rotate_vector(np.array([
                s-s*u/2 ,
                s*sqrt(3)*u/2 ,
                max(0 , h*v-a.value.get_value()*(1-u))
            ]), PI*4/3),
            u_range = [0,1],
            v_range = [0,1],
            checkerboard_colors=[ManimColor('#EBA851'), ManimColor('#F68002')]
        ))
        panel6 = always_redraw(lambda : Surface(
            lambda u,v : rotate_vector(np.array([
                s-s*u/2 ,
                s*sqrt(3)*u/2 ,
                max(0 , h*v-a.value.get_value()*u)
            ]), PI*5/3),
            u_range = [0,1],
            v_range = [0,1],
            checkerboard_colors=[ManimColor('#EBA851'), ManimColor('#F68002')]
        ))

        self.add(panel1, panel2, panel3, panel4, panel5, panel6)

        bottom1 = Surface(
            lambda u,v : np.array([
                s*(u+v/2-1) ,
                s*sqrt(3)*v/2 ,
                0
            ]),
            u_range = [0,1],
            v_range = [0,1],
            checkerboard_colors=[ManimColor('#EBA851'), ManimColor('#F68002')]
        )
        bottom2 = Surface(
            lambda u,v : rotate_vector(np.array([
                s*(u+v/2-1) ,
                s*sqrt(3)*v/2 ,
                0
            ]), PI*2/3),
            u_range = [0,1],
            v_range = [0,1],
            checkerboard_colors=[ManimColor('#EBA851'), ManimColor('#F68002')]
        )
        bottom3 = Surface(
            lambda u,v : rotate_vector(np.array([
                s*(u+v/2-1) ,
                s*sqrt(3)*v/2 ,
                0
            ]), PI*4/3),
            u_range = [0,1],
            v_range = [0,1],
            checkerboard_colors=[ManimColor('#EBA851'), ManimColor('#F68002')]
        )

        self.add(bottom1, bottom2, bottom3)

        top1 = always_redraw(lambda : Surface(
            lambda u,v : np.array([
                s*(u+v/2-1) ,
                s*sqrt(3)*v/2 ,
                h-a.value.get_value()*(v-u)
            ]),
            u_range = [0,1],
            v_range = [0,1],
            checkerboard_colors=[ManimColor('#EBA851'), ManimColor('#F68002')]
        ))
        top2 = always_redraw(lambda : Surface(
            lambda u,v : rotate_vector(np.array([
                s*(u+v/2-1) ,
                s*sqrt(3)*v/2 ,
                h-a.value.get_value()*(v-u)
            ]), PI*2/3),
            u_range = [0,1],
            v_range = [0,1],
            checkerboard_colors=[ManimColor('#EBA851'), ManimColor('#F68002')]
        ))
        top3 = always_redraw(lambda : Surface(
            lambda u,v : rotate_vector(np.array([
                s*(u+v/2-1) ,
                s*sqrt(3)*v/2 ,
                h-a.value.get_value()*(v-u)
            ]), PI*4/3),
            u_range = [0,1],
            v_range = [0,1],
            checkerboard_colors=[ManimColor('#EBA851'), ManimColor('#F68002')]
        ))

        self.add(top1, top2, top3)

        self.begin_ambient_camera_rotation(rate=PI*2/4)
        self.play(a.tracker.animate.set_value(3))
        self.wait(1)
        self.play(a.tracker.animate.set_value(0))
        self.wait(1)

class Slide2_2(Scene):
    def construct(self):
        self.camera.background_color = '#191919'
        p0 = Circle(radius=0.25, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=10).move_to([-6,-1,0]).set_z_index(2)
        p1 = Circle(radius=0.25, color=BLACK, fill_opacity=1, stroke_color=BLUE, stroke_width=10).move_to([6,1,0]).set_z_index(2)
        self.add(p0,p1)
        self.play(Create(CubicBezier([-6,-1,0],[0,-4,0],[0,4,0],[6,1,0], stroke_width=15, fill_color=WHITE).set_z_index(1)), run_time=2)

class Slide3_1(Scene):
    def construct(self):
        self.camera.background_color = '#191919'
        p0 = Circle(radius=0.25, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=10).move_to([-5,-2,0]).set_z_index(2)
        p1 = Circle(radius=0.25, color=BLACK, fill_opacity=1, stroke_color=BLUE, stroke_width=10).move_to([5,2,0]).set_z_index(2)
        l0 = MathTex("A", color=RED).move_to([-5,-1.3,0])
        l1 = MathTex("B", color=BLUE).move_to([5,2.7,0])
        self.add(p0,p1,l0,l1)

class Slide3_4(Scene):
    def construct(self):
        self.camera.background_color = '#191919'
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

class Slide4_1(Scene):
    def construct(self):
        self.camera.background_color = '#191919'
        p0 = Circle(radius=0.25, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=10).move_to([-6,-2,0]).set_z_index(2)
        p1 = Circle(radius=0.25, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=10).move_to([0,2,0]).set_z_index(2)
        p2 = Circle(radius=0.25, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=10).move_to([6,0,0]).set_z_index(2)
        label0 = MathTex("P_0", color=RED).move_to([-6,-1.3,0])
        label1 = MathTex("P_1", color=RED).move_to([0,2.7,0])
        label2 = MathTex("P_2", color=RED).move_to([6,0.7,0])
        self.add(p0,p1,p2,label0,label1,label2)

        l0 = Line(
                p0.get_center(),
                p1.get_center(),
                stroke_width=15
            )
        l1 = Line(
                p1.get_center(),
                p2.get_center(),
                stroke_width=15
            )
        lines = VGroup(l0,l1)
        
        self.play(
            Create(lines),
            run_time=2
        )

class Slide4_7_1(Scene):
    def construct(self):
        self.camera.background_color = '#191919'
        p0 = Circle(radius=0.25, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=10).move_to([-6.5,2.5,0]).set_z_index(2)
        p1 = Circle(radius=0.25, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=10).move_to([3.5,2.5,0]).set_z_index(2)
        r = ParametricFunction(lambda t : np.array([t**5*5/32-1.5 , t**2-1.5 , 0]), t_range=(-2,2), stroke_color=WHITE, stroke_width=15) 
        self.add(p0, p1, r)

        t = Variable(var=-2, label='', num_decimal_places=2).move_to([0,20,0])

        rP = always_redraw(lambda : Circle(
            radius=0.15, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=7
        ).set_z_index(5).move_to(np.array([t.value.get_value()**5*5/32-1.5 , t.value.get_value()**2-1.5 , 0])))

        rPrime = always_redraw(lambda : Arrow(
            np.array([t.value.get_value()**5*5/32-1.5 , t.value.get_value()**2-1.5 , 0]),
            np.array([t.value.get_value()**5*5/32-1.5 , t.value.get_value()**2-1.5 , 0]) + np.array([5/32*5*(t.value.get_value()**4) , 2*t.value.get_value() , 0])/4,
            buff=0,
            color=GREEN,
            stroke_width=25
        ).set_z_index(4))

        self.add(rP, rPrime, t)

        self.play(t.tracker.animate.set_value(-2), run_time=0.3)
        self.play(t.tracker.animate.set_value(2), run_time=4, rate_func=linear)
        self.play(t.tracker.animate.set_value(2), run_time=0.1)

class Slide4_7_2(Scene):
    def construct(self):
        self.camera.background_color = '#191919'
        p0 = Circle(radius=0.25, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=10).move_to([-6,2.5,0]).set_z_index(2)
        p1 = Circle(radius=0.25, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=10).move_to([4,2.5,0]).set_z_index(2)
        r = ParametricFunction(lambda t : np.array([t**5*5/32-1 , t**2-1.5 , 0]), t_range=(-2,2), stroke_color=WHITE, stroke_width=15) 
        self.add(p0, p1, r)

        t = Variable(var=-2, label='', num_decimal_places=2).move_to([0,20,0])

        rP = always_redraw(lambda : Circle(
            radius=0.15, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=7
        ).set_z_index(5).move_to(np.array([t.value.get_value()**5*5/32-1 , t.value.get_value()**2-1.5 , 0])))

        rT = always_redraw(lambda : Arrow(
            np.array([t.value.get_value()**5*5/32-1 , t.value.get_value()**2-1.5 , 0]),
            np.array([t.value.get_value()**5*5/32-1 , t.value.get_value()**2-1.5 , 0]) + ( np.array([5/32*5*(t.value.get_value()**4) , 2*t.value.get_value() , 0]) ) * (0 if np.linalg.norm( np.array([5/32*5*(t.value.get_value()**4) , 2*t.value.get_value() , 0]) ) == 0 else 2.4/np.linalg.norm( np.array([5/32*5*(t.value.get_value()**4) , 2*t.value.get_value() , 0]) )),
            buff=0,
            color=BLUE,
            stroke_width=25
        ).set_z_index(3))

        self.add(rP, rT, t)

        self.play(t.tracker.animate.set_value(-2), run_time=0.3)
        self.play(t.tracker.animate.set_value(2), run_time=4, rate_func=linear)
        self.play(t.tracker.animate.set_value(2), run_time=0.1)

class Slide4_8(Scene):
    def Lerp(self , starting_point , ending_point , t , flag=0):
        return [int(a*(1-t)+b*t) for a, b in zip(starting_point, ending_point)] if flag else [a*(1-t)+b*t for a, b in zip(starting_point, ending_point)]
    
    def construct(self):
        self.camera.background_color = '#191919'
        p0 = Circle(radius=0.25, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=10).move_to([-4,-3,0]).set_z_index(2)
        p1 = Circle(radius=0.25, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=10).move_to([0,1,0]).set_z_index(2)
        p2 = Circle(radius=0.25, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=10).move_to([4,-1,0]).set_z_index(2)

        self.add(p0, p1, p2)

        l0 = Line(
                p0.get_center(),
                p1.get_center(),
                stroke_width=15
        )
        l1 = Line(
                p1.get_center(),
                p2.get_center(),
                stroke_width=15
        )

        self.add(l0, l1)

        t = Variable(var=0, label='', num_decimal_places=2).move_to([0,20,0])

        r = always_redraw(lambda : Circle(
            radius=0.15, color=BLACK, fill_opacity=1, stroke_color=GREEN, stroke_width=7
        ).set_z_index(5).move_to(
            self.Lerp( p0.get_center() , p1.get_center() , t.value.get_value() ) if t.value.get_value()<1 else self.Lerp( p1.get_center() , p2.get_center() , t.value.get_value()-1 )
        ))
        rPrime = always_redraw(lambda : Arrow(
            r.get_center(),
            r.get_center() + 1/2*( p1.get_center()-p0.get_center() if t.value.get_value()<1 else p2.get_center()-p1.get_center() ),
            buff=0,
            color=BLUE,
            stroke_width=25
        ).set_z_index(4))

        self.add(r, rPrime, t)

        self.play(t.tracker.animate.set_value(0), run_time=0.3)
        self.play(t.tracker.animate.set_value(2), run_time=2, rate_func=linear)
        self.play(t.tracker.animate.set_value(2), run_time=0.1)

class Slide6_1(Scene):
    def Hermite(self, t, p):
        p0, v0, p1, v1 = p
        h00 = 2*t**3 - 3*t**2 + 1
        h10 = t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2
        h11 = t**3 - t**2
        return h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1

    def construct(self):
        self.camera.background_color = '#191919'
        p0 = Circle(radius=0.25, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=10).move_to([-6,3,0]).set_z_index(3)
        v0 = Arrow(
            p0.get_center(),
            p0.get_center() + np.array([1,-6,0]),
            buff=0,
            color=BLUE,
            stroke_width=10
        ).set_z_index(2)
        p1 = Circle(radius=0.25, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=10).move_to([2,2,0]).set_z_index(3)
        v1 = Arrow(
            p1.get_center(),
            p1.get_center() + np.array([4,-5,0]),
            buff=0,
            color=BLUE,
            stroke_width=10
        ).set_z_index(2)
        
        self.add(p0,v0,p1,v1)

        Lp0 = MathTex("P_0", color=RED).move_to([-5.4,3.2,0])
        Lv0 = MathTex("v_0", color=BLUE).move_to([-4.75,-3.25,0])
        Lp1 = MathTex("P_1", color=RED).move_to([2.6,2.2,0])
        Lv1 = MathTex("v_1", color=BLUE).move_to([6.25,-3.25,0])

        self.add(Lp0,Lv0,Lp1,Lv1)

        curve = ParametricFunction(
            lambda t: self.Hermite(
                t,
                [
                    p0.get_center(),
                    np.array([1,-6,0]),
                    p1.get_center(),
                    np.array([4,-5,0])
                ]
            ),
            t_range = (0, 1),
            color = WHITE,
            stroke_width=15
        )
        q = MathTex("?").scale(3).move_to([-2,1.5,0])
        self.play(Create(DashedVMobject(curve, num_dashes=10)), run_time=2)
        self.play(FadeIn(q))

class Slide6_8(Scene):
    def Hermite(self, t, p):
        p0, v0, p1, v1 = p
        h00 = 2*t**3 - 3*t**2 + 1
        h10 = t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2
        h11 = t**3 - t**2
        return h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1
    
    def HermitePrime(self, t, p):
        p0, v0, p1, v1 = p
        h00 = 6*t**2 - 6*t
        h10 = 3*t**2 - 4*t + 1
        h01 = -6*t**2 + 6*t
        h11 = 3*t**2 - 2*t
        return h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1

    def construct(self):
        self.camera.background_color = '#191919'
        p0 = Circle(radius=0.25, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=10).move_to([-6,3,0]).set_z_index(3)
        v0 = Arrow(
            p0.get_center(),
            p0.get_center() + np.array([1,-6,0]),
            buff=0,
            color=BLUE,
            stroke_width=10
        ).set_z_index(2)
        p1 = Circle(radius=0.25, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=10).move_to([2,2,0]).set_z_index(3)
        v1 = Arrow(
            p1.get_center(),
            p1.get_center() + np.array([4,-5,0]),
            buff=0,
            color=BLUE,
            stroke_width=10
        ).set_z_index(2)
        
        self.add(p0,v0,p1,v1)

        Lp0 = MathTex("P_0", color=RED).move_to([-5.4,3.2,0])
        Lv0 = MathTex("v_0", color=BLUE).move_to([-4.75,-3.25,0])
        Lp1 = MathTex("P_1", color=RED).move_to([2.6,2.2,0])
        Lv1 = MathTex("v_1", color=BLUE).move_to([6.25,-3.25,0])

        self.add(Lp0,Lv0,Lp1,Lv1)

        curve = ParametricFunction(
            lambda t: self.Hermite(
                t,
                [
                    p0.get_center(),
                    np.array([1,-6,0]),
                    p1.get_center(),
                    np.array([4,-5,0])
                ]
            ),
            t_range = (0, 1),
            color = WHITE,
            stroke_width=15
        )
        self.add(curve)

        t = Variable(var=0, label='', num_decimal_places=2).move_to([0,20,0])

        r = always_redraw(lambda : Circle(
            radius=0.15, color=BLACK, fill_opacity=1, stroke_color=GREEN, stroke_width=7
        ).set_z_index(5).move_to(
            self.Hermite(t.value.get_value(),[
                    p0.get_center(),
                    np.array([1,-6,0]),
                    p1.get_center(),
                    np.array([4,-5,0])
                ])
        ))
        rPrime = always_redraw(lambda : Arrow(
            r.get_center(),
            r.get_center() + 3*self.HermitePrime(t.value.get_value(),[
                    p0.get_center(),
                    np.array([1,-6,0]),
                    p1.get_center(),
                    np.array([4,-5,0])
                ])/np.linalg.norm(self.HermitePrime(t.value.get_value(),[
                    p0.get_center(),
                    np.array([1,-6,0]),
                    p1.get_center(),
                    np.array([4,-5,0])
                ])),
            buff=0,
            color=GREEN,
            stroke_width=15
        ).set_z_index(4))

        self.add(r, rPrime, t)

        self.play(t.tracker.animate.set_value(0), run_time=0.3)
        self.play(t.tracker.animate.set_value(1), run_time=2, rate_func=linear)
        self.play(t.tracker.animate.set_value(1), run_time=0.1)

class Slide6_9(Scene):
    def construct(self):
        self.camera.background_color = '#191919'
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

        f1 = ParametricFunction(
            lambda t : ax.coords_to_point(t,1-3*t**2+2*t**3),
            t_range = (0, 1),
            color = RED,
        )
        f2 = ParametricFunction(
            lambda t : ax.coords_to_point(t,t-2*t**2+t**3),
            t_range = (0, 1),
            color = GREEN,
        )
        f3 = ParametricFunction(
            lambda t : ax.coords_to_point(t,3*t**2-2*t**3),
            t_range = (0, 1),
            color = BLUE,
        )
        f4 = ParametricFunction(
            lambda t : ax.coords_to_point(t,-t**2+t**3),
            t_range = (0, 1),
            color = YELLOW,
        )

        self.play(
            Create(f1),
            Create(f2),
            Create(f3),
            Create(f4),
            run_time=2
        )

class Slide6_10(Scene):
    def Hermite(self, t, p):
        p0, v0, p1, v1 = p
        h00 = 2*t**3 - 3*t**2 + 1
        h10 = t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2
        h11 = t**3 - t**2
        return h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1
    
    def HermitePrime(self, t, p):
        p0, v0, p1, v1 = p
        h00 = 6*t**2 - 6*t
        h10 = 3*t**2 - 4*t + 1
        h01 = -6*t**2 + 6*t
        h11 = 3*t**2 - 2*t
        return h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1

    def construct(self):
        self.camera.background_color = '#191919'
        ax = NumberPlane(
            x_range=[-24, 24, 1],
            y_range=[-12, 12, 1],
            x_length=12,
            y_length=6,
        )

        P0 = ax.c2p(-25,-10,0)
        V0 = ax.c2p(10,25,0)
        P1 = ax.c2p(-20,10,0)
        V1 = ax.c2p(30,-10,0)
        P2 = ax.c2p(-10,5,0)
        V2 = ax.c2p(10,-5,0)
        P3 = ax.c2p(-5,-13,0)
        V3 = ax.c2p(20,5,0)
        P4 = ax.c2p(22,13,0)
        V4 = ax.c2p(-24,-2,0)
        P5 = ax.c2p(0,7,0)
        V5 = ax.c2p(-7,8,0)

        P = np.array([P0,P1,P2,P3,P4,P5])
        V = np.array([V0,V1,V2,V3,V4,V5])

        p0 = Circle(radius=0.125, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=5).move_to(P0).set_z_index(3)
        v0 = Arrow(
            P0,
            P0 + V0,
            buff=0,
            color=BLUE,
            stroke_width=5
        ).set_z_index(2)
        p1 = Circle(radius=0.125, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=5).move_to(P1).set_z_index(3)
        v1 = Arrow(
            P1,
            P1 + V1,
            buff=0,
            color=BLUE,
            stroke_width=5
        ).set_z_index(2)
        p2 = Circle(radius=0.125, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=5).move_to(P2).set_z_index(3)
        v2 = Arrow(
            P2,
            P2 + V2,
            buff=0,
            color=BLUE,
            stroke_width=5
        ).set_z_index(2)
        p3 = Circle(radius=0.125, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=5).move_to(P3).set_z_index(3)
        v3 = Arrow(
            P3,
            P3 + V3,
            buff=0,
            color=BLUE,
            stroke_width=5
        ).set_z_index(2)
        p4 = Circle(radius=0.125, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=5).move_to(P4).set_z_index(3)
        v4 = Arrow(
            P4,
            P4 + V4,
            buff=0,
            color=BLUE,
            stroke_width=5
        ).set_z_index(2)
        p5 = Circle(radius=0.125, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=5).move_to(P5).set_z_index(3)
        v5 = Arrow(
            P5,
            P5 + V5,
            buff=0,
            color=BLUE,
            stroke_width=5
        ).set_z_index(2)
        
        self.add(p0,v0,p1,v1,p2,v2,p3,v3,p4,v4,p5,v5)

        curve0 = ParametricFunction(
            lambda t: self.Hermite(
                t,
                [
                    P0,
                    V0,
                    P1,
                    V1
                ]
            ),
            t_range = (0, 1),
            color = WHITE,
            stroke_width=7.5
        )
        curve1 = ParametricFunction(
            lambda t: self.Hermite(
                t,
                [
                    P1,
                    V1,
                    P2,
                    V2
                ]
            ),
            t_range = (0, 1),
            color = WHITE,
            stroke_width=7.5
        )
        curve2 = ParametricFunction(
            lambda t: self.Hermite(
                t,
                [
                    P2,
                    V2,
                    P3,
                    V3
                ]
            ),
            t_range = (0, 1),
            color = WHITE,
            stroke_width=7.5
        )
        curve3 = ParametricFunction(
            lambda t: self.Hermite(
                t,
                [
                    P3,
                    V3,
                    P4,
                    V4
                ]
            ),
            t_range = (0, 1),
            color = WHITE,
            stroke_width=7.5
        )
        curve4 = ParametricFunction(
            lambda t: self.Hermite(
                t,
                [
                    P4,
                    V4,
                    P5,
                    V5
                ]
            ),
            t_range = (0, 1),
            color = WHITE,
            stroke_width=7.5
        )
        self.add(curve0, curve1, curve2, curve3, curve4)

        t = Variable(var=0, label='', num_decimal_places=2).move_to([0,20,0])

        r = always_redraw(lambda : Circle(
            radius=0.075, color=BLACK, fill_opacity=1, stroke_color=GREEN, stroke_width=3.5
        ).set_z_index(5).move_to(
            self.Hermite(t.value.get_value()%1,[
                    P[floor(t.value.get_value())],
                    V[floor(t.value.get_value())],
                    P[floor(t.value.get_value())+1],
                    V[floor(t.value.get_value())+1]
                ])
            if t.value.get_value()<5 else P[5]
        ))
        
        
        rPrime = always_redraw(lambda : Arrow(
            r.get_center(),
            r.get_center() + 1/5*(self.HermitePrime(t.value.get_value()%1,[
                    P[floor(t.value.get_value())],
                    V[floor(t.value.get_value())],
                    P[floor(t.value.get_value())+1],
                    V[floor(t.value.get_value())+1]
                ]) if t.value.get_value()<5 else V[5]),
            buff=0,
            color=GREEN,
            stroke_width=7.5
        ).set_z_index(4))
        

        self.add(r, rPrime, t)

        self.play(t.tracker.animate.set_value(0), run_time=0.3)
        self.play(t.tracker.animate.set_value(5), run_time=5, rate_func=linear)
        self.play(t.tracker.animate.set_value(5), run_time=0.1)

class Slide6_11(Scene):
    def Hermite(self, t, p):
        p0, v0, p1, v1 = p
        h00 = 2*t**3 - 3*t**2 + 1
        h10 = t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2
        h11 = t**3 - t**2
        return h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1
    
    def HermitePrime(self, t, p):
        p0, v0, p1, v1 = p
        h00 = 6*t**2 - 6*t
        h10 = 3*t**2 - 4*t + 1
        h01 = -6*t**2 + 6*t
        h11 = 3*t**2 - 2*t
        return h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1
    
    def HermitePPrime(self, t, p):
        p0, v0, p1, v1 = p
        h00 = 12*t - 6
        h10 = 6*t - 4
        h01 = -12*t + 6
        h11 = 6*t - 2
        return h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1

    def construct(self):
        self.camera.background_color = '#191919'
        ax = NumberPlane(
            x_range=[-24, 24, 1],
            y_range=[-12, 12, 1],
            x_length=12,
            y_length=6,
        )

        P0 = ax.c2p(-25,-10,0)
        V0 = ax.c2p(10,25,0)
        P1 = ax.c2p(-20,10,0)
        V1 = ax.c2p(30,-10,0)
        P2 = ax.c2p(-10,5,0)
        V2 = ax.c2p(10,-5,0)
        P3 = ax.c2p(-5,-13,0)
        V3 = ax.c2p(20,5,0)
        P4 = ax.c2p(22,13,0)
        V4 = ax.c2p(-24,-2,0)
        P5 = ax.c2p(0,7,0)
        V5 = ax.c2p(-7,8,0)

        P = np.array([P0,P1,P2,P3,P4,P5])
        V = np.array([V0,V1,V2,V3,V4,V5])

        p0 = Circle(radius=0.125, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=5).move_to(P0).set_z_index(3)
        v0 = Arrow(
            P0,
            P0 + V0,
            buff=0,
            color=BLUE,
            stroke_width=5
        ).set_z_index(2).set_opacity(0.5)
        p1 = Circle(radius=0.125, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=5).move_to(P1).set_z_index(3)
        v1 = Arrow(
            P1,
            P1 + V1,
            buff=0,
            color=BLUE,
            stroke_width=5
        ).set_z_index(2).set_opacity(0.5)
        p2 = Circle(radius=0.125, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=5).move_to(P2).set_z_index(3)
        v2 = Arrow(
            P2,
            P2 + V2,
            buff=0,
            color=BLUE,
            stroke_width=5
        ).set_z_index(2).set_opacity(0.5)
        p3 = Circle(radius=0.125, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=5).move_to(P3).set_z_index(3)
        v3 = Arrow(
            P3,
            P3 + V3,
            buff=0,
            color=BLUE,
            stroke_width=5
        ).set_z_index(2).set_opacity(0.5)
        p4 = Circle(radius=0.125, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=5).move_to(P4).set_z_index(3)
        v4 = Arrow(
            P4,
            P4 + V4,
            buff=0,
            color=BLUE,
            stroke_width=5
        ).set_z_index(2).set_opacity(0.5)
        p5 = Circle(radius=0.125, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=5).move_to(P5).set_z_index(3)
        v5 = Arrow(
            P5,
            P5 + V5,
            buff=0,
            color=BLUE,
            stroke_width=5
        ).set_z_index(2).set_opacity(0.5)
        
        self.add(p0,v0,p1,v1,p2,v2,p3,v3,p4,v4,p5,v5)

        curve0 = ParametricFunction(
            lambda t: self.Hermite(
                t,
                [
                    P0,
                    V0,
                    P1,
                    V1
                ]
            ),
            t_range = (0, 1),
            color = WHITE,
            stroke_width=7.5
        )
        curve1 = ParametricFunction(
            lambda t: self.Hermite(
                t,
                [
                    P1,
                    V1,
                    P2,
                    V2
                ]
            ),
            t_range = (0, 1),
            color = WHITE,
            stroke_width=7.5
        )
        curve2 = ParametricFunction(
            lambda t: self.Hermite(
                t,
                [
                    P2,
                    V2,
                    P3,
                    V3
                ]
            ),
            t_range = (0, 1),
            color = WHITE,
            stroke_width=7.5
        )
        curve3 = ParametricFunction(
            lambda t: self.Hermite(
                t,
                [
                    P3,
                    V3,
                    P4,
                    V4
                ]
            ),
            t_range = (0, 1),
            color = WHITE,
            stroke_width=7.5
        )
        curve4 = ParametricFunction(
            lambda t: self.Hermite(
                t,
                [
                    P4,
                    V4,
                    P5,
                    V5
                ]
            ),
            t_range = (0, 1),
            color = WHITE,
            stroke_width=7.5
        )
        self.add(curve0, curve1, curve2, curve3, curve4)

        t = Variable(var=0, label='', num_decimal_places=2).move_to([0,20,0])

        r = always_redraw(lambda : Circle(
            radius=0.075, color=BLACK, fill_opacity=1, stroke_color=GREEN, stroke_width=3.5
        ).set_z_index(5).move_to(
            self.Hermite(t.value.get_value()%1,[
                    P[floor(t.value.get_value())],
                    V[floor(t.value.get_value())],
                    P[floor(t.value.get_value())+1],
                    V[floor(t.value.get_value())+1]
                ])
            if t.value.get_value()<5 else P[5]
        ))
        
        
        rPrime = always_redraw(lambda : Arrow(
            r.get_center(),
            r.get_center() + 1/5*(self.HermitePrime(t.value.get_value()%1,[
                    P[floor(t.value.get_value())],
                    V[floor(t.value.get_value())],
                    P[floor(t.value.get_value())+1],
                    V[floor(t.value.get_value())+1]
                ]) if t.value.get_value()<5 else V[5]),
            buff=0,
            color=GREEN,
            stroke_width=7.5
        ).set_z_index(4))

        rPPrime = always_redraw(lambda : Arrow(
            rPrime.get_end(),
            rPrime.get_end() + 1/15*(self.HermitePPrime(t.value.get_value()%1,[
                    P[floor(t.value.get_value())],
                    V[floor(t.value.get_value())],
                    P[floor(t.value.get_value())+1],
                    V[floor(t.value.get_value())+1]
                ]) if t.value.get_value()<5 else self.HermitePPrime(1,[
                    P[4],
                    V[4],
                    P[5],
                    V[5]
                ])),
            buff=0,
            color=RED,
            stroke_width=7.5
        ).set_z_index(3))
        

        self.add(r, rPrime, rPPrime, t)

        self.play(t.tracker.animate.set_value(0), run_time=0.3)
        self.play(t.tracker.animate.set_value(5), run_time=5, rate_func=linear)
        self.play(t.tracker.animate.set_value(5), run_time=0.1)

class Slide6_12(Scene):
    def Hermite(self, t, p):
        p0, v0, p1, v1 = p
        h00 = 2*t**3 - 3*t**2 + 1
        h10 = t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2
        h11 = t**3 - t**2
        return h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1
    
    def HermitePrime(self, t, p):
        p0, v0, p1, v1 = p
        h00 = 6*t**2 - 6*t
        h10 = 3*t**2 - 4*t + 1
        h01 = -6*t**2 + 6*t
        h11 = 3*t**2 - 2*t
        return h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1
    
    def HermitePPrime(self, t, p):
        p0, v0, p1, v1 = p
        h00 = 12*t - 6
        h10 = 6*t - 4
        h01 = -12*t + 6
        h11 = 6*t - 2
        return h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1
    
    def Curvature(self, rP, rPP):
        return np.linalg.norm(np.cross(rP,rPP))/(np.linalg.norm(rP)**3)

    def construct(self):
        self.camera.background_color = '#191919'
        ax = NumberPlane(
            x_range=[-24, 24, 1],
            y_range=[-12, 12, 1],
            x_length=6,
            y_length=3,
        )

        P0 = ax.c2p(-25,-10,0)
        V0 = ax.c2p(10,25,0)
        P1 = ax.c2p(-20,10,0)
        V1 = ax.c2p(30,-10,0)
        P2 = ax.c2p(-10,5,0)
        V2 = ax.c2p(10,-5,0)
        P3 = ax.c2p(-5,-13,0)
        V3 = ax.c2p(20,5,0)
        P4 = ax.c2p(22,13,0)
        V4 = ax.c2p(-24,-2,0)
        P5 = ax.c2p(0,7,0)
        V5 = ax.c2p(-7,8,0)

        P = np.array([P0,P1,P2,P3,P4,P5])
        V = np.array([V0,V1,V2,V3,V4,V5])

        p0 = Circle(radius=0.0625, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=2.5).move_to(P0).set_z_index(3)
        v0 = Arrow(
            P0,
            P0 + V0,
            buff=0,
            color=BLUE,
            stroke_width=2.5,
            tip_length=0.15
        ).set_z_index(2).set_opacity(0.5)
        p1 = Circle(radius=0.0625, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=2.5).move_to(P1).set_z_index(3)
        v1 = Arrow(
            P1,
            P1 + V1,
            buff=0,
            color=BLUE,
            stroke_width=2.5,
            tip_length=0.15
        ).set_z_index(2).set_opacity(0.5)
        p2 = Circle(radius=0.0625, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=2.5).move_to(P2).set_z_index(3)
        v2 = Arrow(
            P2,
            P2 + V2,
            buff=0,
            color=BLUE,
            stroke_width=2.5,
            tip_length=0.15
        ).set_z_index(2).set_opacity(0.5)
        p3 = Circle(radius=0.0625, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=2.5).move_to(P3).set_z_index(3)
        v3 = Arrow(
            P3,
            P3 + V3,
            buff=0,
            color=BLUE,
            stroke_width=2.5,
            tip_length=0.15
        ).set_z_index(2).set_opacity(0.5)
        p4 = Circle(radius=0.0625, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=2.5).move_to(P4).set_z_index(3)
        v4 = Arrow(
            P4,
            P4 + V4,
            buff=0,
            color=BLUE,
            stroke_width=2.5,
            tip_length=0.15
        ).set_z_index(2).set_opacity(0.5)
        p5 = Circle(radius=0.0625, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=2.5).move_to(P5).set_z_index(3)
        v5 = Arrow(
            P5,
            P5 + V5,
            buff=0,
            color=BLUE,
            stroke_width=2.5,
            tip_length=0.15
        ).set_z_index(2).set_opacity(0.5)
        
        self.add(p0,v0,p1,v1,p2,v2,p3,v3,p4,v4,p5,v5)

        curve0 = ParametricFunction(
            lambda t: self.Hermite(
                t,
                [
                    P0,
                    V0,
                    P1,
                    V1
                ]
            ),
            t_range = (0, 1),
            color = WHITE,
            stroke_width=3.75
        )
        curve1 = ParametricFunction(
            lambda t: self.Hermite(
                t,
                [
                    P1,
                    V1,
                    P2,
                    V2
                ]
            ),
            t_range = (0, 1),
            color = WHITE,
            stroke_width=3.75
        )
        curve2 = ParametricFunction(
            lambda t: self.Hermite(
                t,
                [
                    P2,
                    V2,
                    P3,
                    V3
                ]
            ),
            t_range = (0, 1),
            color = WHITE,
            stroke_width=3.75
        )
        curve3 = ParametricFunction(
            lambda t: self.Hermite(
                t,
                [
                    P3,
                    V3,
                    P4,
                    V4
                ]
            ),
            t_range = (0, 1),
            color = WHITE,
            stroke_width=3.75
        )
        curve4 = ParametricFunction(
            lambda t: self.Hermite(
                t,
                [
                    P4,
                    V4,
                    P5,
                    V5
                ]
            ),
            t_range = (0, 1),
            color = WHITE,
            stroke_width=3.75
        )
        self.add(curve0, curve1, curve2, curve3, curve4)

        t = Variable(var=0, label='', num_decimal_places=2).move_to([0,20,0])

        r = always_redraw(lambda : Circle(
            radius=0.0375, color=BLACK, fill_opacity=1, stroke_color=GREEN, stroke_width=1.75
        ).set_z_index(5).move_to(
            self.Hermite(t.value.get_value()%1,[
                    P[floor(t.value.get_value())],
                    V[floor(t.value.get_value())],
                    P[floor(t.value.get_value())+1],
                    V[floor(t.value.get_value())+1]
                ])
            if t.value.get_value()<5 else P[5]
        ))
        
        
        rPrime = always_redraw(lambda : Arrow(
            r.get_center(),
            r.get_center() + 1/5*(self.HermitePrime(t.value.get_value()%1,[
                    P[floor(t.value.get_value())],
                    V[floor(t.value.get_value())],
                    P[floor(t.value.get_value())+1],
                    V[floor(t.value.get_value())+1]
                ]) if t.value.get_value()<5 else V[5]),
            buff=0,
            color=GREEN,
            stroke_width=3.75
        ).set_z_index(4))

        
        
        normal_vector = lambda t_val : normalize(
            np.array([
                -(self.HermitePrime(t_val%1,[
                    P[floor(t_val)],
                    V[floor(t_val)],
                    P[floor(t_val)+1],
                    V[floor(t_val)+1]
                ])[1] if t_val<5 else V[5][1]),
                (self.HermitePrime(t_val%1,[
                    P[floor(t_val)],
                    V[floor(t_val)],
                    P[floor(t_val)+1],
                    V[floor(t_val)+1]
                ])[0] if t_val<5 else V[5][0]),
                0
            ])
        ) * ( 1 if np.cross(
            (self.HermitePrime(t_val%1,[
                    P[floor(t_val)],
                    V[floor(t_val)],
                    P[floor(t_val)+1],
                    V[floor(t_val)+1]
                ]) if t_val<5 else V[5]),
            (self.HermitePPrime(t_val%1,[
                    P[floor(t_val)],
                    V[floor(t_val)],
                    P[floor(t_val)+1],
                    V[floor(t_val)+1]
                ]) if t_val<5 else V[5])
                )[2]>0 else -1 )

        curvature = lambda t_val : (
            np.linalg.norm(np.cross(
                self.HermitePrime(t_val%1,[
                    P[floor(t_val)],
                    V[floor(t_val)],
                    P[floor(t_val)+1],
                    V[floor(t_val)+1]
                ]) if t_val<5 else V[5],
                self.HermitePPrime(t_val%1,[
                    P[floor(t_val)],
                    V[floor(t_val)],
                    P[floor(t_val)+1],
                    V[floor(t_val)+1]
                ]) if t_val<5 else self.HermitePPrime(1,[
                    P[4],
                    V[4],
                    P[5],
                    V[5]
                ])
            )) /
            (np.linalg.norm(
                self.HermitePrime(t_val%1,[
                    P[floor(t_val)],
                    V[floor(t_val)],
                    P[floor(t_val)+1],
                    V[floor(t_val)+1]
                ]) if t_val<5 else V[5]
            )**3)
        )

        osc_circle = always_redraw(lambda : 
            Circle(
                radius=1/curvature(t.value.get_value()),
                color=WHITE,
                stroke_width=2.5
            ).move_to(
                r.get_center() + (1/curvature(t.value.get_value())) * normal_vector(t.value.get_value())
            ).set_z_index(2)
        )

        rN = always_redraw(lambda : Arrow(
            r.get_center(),
            r.get_center() + (1/curvature(t.value.get_value())) * normal_vector(t.value.get_value()),
            buff=0,
            color=WHITE,
            stroke_width=2.5,
            tip_length=0.15
        ).set_z_index(4))

        

        self.add(r, rPrime, osc_circle, rN, t)

        self.play(t.tracker.animate.set_value(0), run_time=0.3)
        self.play(t.tracker.animate.set_value(5), run_time=10, rate_func=linear)
        self.play(t.tracker.animate.set_value(5), run_time=0.1)

class Slide6_13(Scene):
    def Hermite(self, t, p):
        p0, v0, p1, v1 = p
        h00 = 2*t**3 - 3*t**2 + 1
        h10 = t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2
        h11 = t**3 - t**2
        return h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1
    
    def HermitePrime(self, t, p):
        p0, v0, p1, v1 = p
        h00 = 6*t**2 - 6*t
        h10 = 3*t**2 - 4*t + 1
        h01 = -6*t**2 + 6*t
        h11 = 3*t**2 - 2*t
        return h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1
    
    def HermitePPrime(self, t, p):
        p0, v0, p1, v1 = p
        h00 = 12*t - 6
        h10 = 6*t - 4
        h01 = -12*t + 6
        h11 = 6*t - 2
        return h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1

    def construct(self):
        self.camera.background_color = '#191919'
        ax = NumberPlane(
            x_range=[-24, 24, 1],
            y_range=[-12, 12, 1],
            x_length=8,
            y_length=4,
        )

        P0 = ax.c2p(-25,-10,0)
        V0 = ax.c2p(10,25,0)
        P1 = ax.c2p(-20,10,0)
        V1 = ax.c2p(30,-10,0)
        P2 = ax.c2p(-10,5,0)
        V2 = ax.c2p(10,-5,0)
        P3 = ax.c2p(-5,-13,0)
        V3 = ax.c2p(20,5,0)
        P4 = ax.c2p(22,13,0)
        V4 = ax.c2p(-24,-2,0)
        P5 = ax.c2p(0,7,0)
        V5 = ax.c2p(-7,8,0)

        P = np.array([P0,P1,P2,P3,P4,P5])
        V = np.array([V0,V1,V2,V3,V4,V5])

        p0 = Circle(radius=0.125, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=5).move_to(P0).set_z_index(3)
        v0 = Arrow(
            P0,
            P0 + V0,
            buff=0,
            color=BLUE,
            stroke_width=5
        ).set_z_index(2).set_opacity(0.5)
        p1 = Circle(radius=0.125, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=5).move_to(P1).set_z_index(3)
        v1 = Arrow(
            P1,
            P1 + V1,
            buff=0,
            color=BLUE,
            stroke_width=5
        ).set_z_index(2).set_opacity(0.5)
        p2 = Circle(radius=0.125, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=5).move_to(P2).set_z_index(3)
        v2 = Arrow(
            P2,
            P2 + V2,
            buff=0,
            color=BLUE,
            stroke_width=5
        ).set_z_index(2).set_opacity(0.5)
        p3 = Circle(radius=0.125, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=5).move_to(P3).set_z_index(3)
        v3 = Arrow(
            P3,
            P3 + V3,
            buff=0,
            color=BLUE,
            stroke_width=5
        ).set_z_index(2).set_opacity(0.5)
        p4 = Circle(radius=0.125, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=5).move_to(P4).set_z_index(3)
        v4 = Arrow(
            P4,
            P4 + V4,
            buff=0,
            color=BLUE,
            stroke_width=5
        ).set_z_index(2).set_opacity(0.5)
        p5 = Circle(radius=0.125, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=5).move_to(P5).set_z_index(3)
        v5 = Arrow(
            P5,
            P5 + V5,
            buff=0,
            color=BLUE,
            stroke_width=5
        ).set_z_index(2).set_opacity(0.5)
        
        self.add(p0,v0,p1,v1,p2,v2,p3,v3,p4,v4,p5,v5)

        curve0 = ParametricFunction(
            lambda t: self.Hermite(
                t,
                [
                    P0,
                    V0,
                    P1,
                    V1
                ]
            ),
            t_range = (0, 1),
            color = WHITE,
            stroke_width=7.5
        )
        curve1 = ParametricFunction(
            lambda t: self.Hermite(
                t,
                [
                    P1,
                    V1,
                    P2,
                    V2
                ]
            ),
            t_range = (0, 1),
            color = WHITE,
            stroke_width=7.5
        )
        curve2 = ParametricFunction(
            lambda t: self.Hermite(
                t,
                [
                    P2,
                    V2,
                    P3,
                    V3
                ]
            ),
            t_range = (0, 1),
            color = WHITE,
            stroke_width=7.5
        )
        curve3 = ParametricFunction(
            lambda t: self.Hermite(
                t,
                [
                    P3,
                    V3,
                    P4,
                    V4
                ]
            ),
            t_range = (0, 1),
            color = WHITE,
            stroke_width=7.5
        )
        curve4 = ParametricFunction(
            lambda t: self.Hermite(
                t,
                [
                    P4,
                    V4,
                    P5,
                    V5
                ]
            ),
            t_range = (0, 1),
            color = WHITE,
            stroke_width=7.5
        )
        self.add(curve0, curve1, curve2, curve3, curve4)




        n_spikes = 200
        n_comb_points = 5000

        t_samples_spikes = np.linspace(0, 5, n_spikes)
        t_samples_comb = np.linspace(0, 5, n_comb_points)

        spikes = VGroup()
        comb_curves = VGroup()  # Store separate comb pieces

        # Create spikes (all at once, over full t range)
        for t_val in t_samples_spikes:
            if t_val < 5:
                i = floor(t_val)
                local_t = t_val % 1
                rP = self.HermitePrime(local_t, [P[i], V[i], P[i+1], V[i+1]])
                rPP = self.HermitePPrime(local_t, [P[i], V[i], P[i+1], V[i+1]])
                r_pos = self.Hermite(local_t, [P[i], V[i], P[i+1], V[i+1]])
            else:
                rP = V[5]
                rPP = self.HermitePPrime(1, [P[4], V[4], P[5], V[5]])
                r_pos = P[5]

            tangent = rP / np.linalg.norm(rP)
            curvature = np.linalg.norm(np.cross(rP, rPP)) / (np.linalg.norm(rP)**3)

            sign = np.cross(rP, rPP)[2]
            normal = normalize(np.array([-rP[1], rP[0], 0]) if sign < 0 else np.array([rP[1], -rP[0], 0]))

            spike_length = 0.25 * curvature
            spike = Line(
                r_pos,
                r_pos + spike_length * normal,
                buff=0,
                stroke_width=2,
                color=WHITE,
            )
            spikes.add(spike)

        # Now create comb curves separately between knots
        for i in range(5):  # 5 spline segments between P0-P1, P1-P2, ..., P4-P5
            # Only sample t in [i, i+1]
            t_subsamples = np.linspace(i, i+1, n_comb_points//5)

            tip_points = []

            for t_val in t_subsamples:
                local_t = t_val - i  # always between 0 and 1
                rP = self.HermitePrime(local_t, [P[i], V[i], P[i+1], V[i+1]])
                rPP = self.HermitePPrime(local_t, [P[i], V[i], P[i+1], V[i+1]])
                r_pos = self.Hermite(local_t, [P[i], V[i], P[i+1], V[i+1]])

                tangent = rP / np.linalg.norm(rP)
                curvature = np.linalg.norm(np.cross(rP, rPP)) / (np.linalg.norm(rP)**3)

                sign = np.cross(rP, rPP)[2]
                normal = normalize(np.array([-rP[1], rP[0], 0]) if sign < 0 else np.array([rP[1], -rP[0], 0]))

                spike_length = 0.25 * curvature
                tip_point = r_pos + spike_length * normal
                tip_points.append(tip_point)

            # Make a separate comb curve for each segment
            comb_curve = VMobject(color=WHITE, stroke_width=2)
            comb_curve.set_points_as_corners(tip_points)
            comb_curves.add(comb_curve)

        t = Variable(var=0, label='', num_decimal_places=2).move_to([0,20,0])

        r = always_redraw(lambda : Circle(
            radius=0.075, color=BLACK, fill_opacity=1, stroke_color=GREEN, stroke_width=3.5
        ).set_z_index(5).move_to(
            self.Hermite(t.value.get_value()%1,[
                    P[floor(t.value.get_value())],
                    V[floor(t.value.get_value())],
                    P[floor(t.value.get_value())+1],
                    V[floor(t.value.get_value())+1]
                ])
            if t.value.get_value()<5 else P[5]
        ))
        
        self.add(t,r)

        self.play(t.tracker.animate.set_value(0), run_time=0.3)
        self.play(Create(spikes), Create(comb_curves), t.tracker.animate.set_value(5), run_time=5, rate_func=linear)
        self.play(t.tracker.animate.set_value(5), run_time=0.1)

class Slide7_1(Scene):
    def construct(self):
        self.camera.background_color = '#191919'
        p0 = Circle(radius=0.25, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=10).move_to([-6,-3,0]).set_z_index(2)
        p1 = Circle(radius=0.25, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=10).move_to([-3,3,0]).set_z_index(2)
        p2 = Circle(radius=0.25, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=10).move_to([3,3,0]).set_z_index(2)
        p3 = Circle(radius=0.25, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=10).move_to([6,-3,0]).set_z_index(2)
        label0 = MathTex("P_0", color=RED).move_to([-5.5,-3.5,0])
        label1 = MathTex("P_1", color=RED).move_to([-3.5,3.5,0])
        label2 = MathTex("P_2", color=RED).move_to([3.5,3.5,0])
        label3 = MathTex("P_3", color=RED).move_to([5.5,-3.5,0])
        self.add(p0,p1,p2,p3,label0,label1,label2,label3)

        l0 = Line(
                p0.get_center(),
                p1.get_center(),
                stroke_width=15
            )
        l1 = Line(
                p1.get_center(),
                p2.get_center(),
                stroke_width=15
            )
        l2 = Line(
                p2.get_center(),
                p3.get_center(),
                stroke_width=15
            )
        lines = VGroup(l0,l1,l2)
        
        self.play(
            Create(lines),
            run_time=3
        )

class Slide7_7(Scene):
    def construct(self):
        self.camera.background_color = '#191919'
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

        f1 = ParametricFunction(
            lambda t : ax.coords_to_point(t,1-3*t+3*t**2-t**3),
            t_range = (0, 1),
            color = RED,
        )
        f2 = ParametricFunction(
            lambda t : ax.coords_to_point(t,3*t-6*t**2+3*t**3),
            t_range = (0, 1),
            color = GREEN,
        )
        f3 = ParametricFunction(
            lambda t : ax.coords_to_point(t,3*t**2-3*t**3),
            t_range = (0, 1),
            color = BLUE,
        )
        f4 = ParametricFunction(
            lambda t : ax.coords_to_point(t,t**3),
            t_range = (0, 1),
            color = YELLOW,
        )

        self.play(
            Create(f1),
            Create(f2),
            Create(f3),
            Create(f4),
            run_time=2
        )

class Slide7_8(Scene):
    def Bezier(self, t, p):
        p0, p1, p2, p3 = p
        return (1-t)**3 * p0 + 3*(1-t)**2 * t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3

    def BezierPrime(self, t, p):
        p0, p1, p2, p3 = p
        return 3*(1-t)**2*(p1 - p0) + 6*(1-t)*t*(p2 - p1) + 3*t**2*(p3 - p2)
    
    def construct(self):
        self.camera.background_color = '#191919'
        ax = NumberPlane(
            x_range=[-24, 24, 1],
            y_range=[-12, 12, 1],
            x_length=12,
            y_length=6,
        )

        P0 = ax.c2p(-25,-10,0)
        P1 = ax.c2p(-24,8,0)
        P2 = ax.c2p(-20,-8,0)
        P3 = ax.c2p(-23,5,0)
        P4 = ax.c2p(-10,10,0)
        P5 = ax.c2p(5,5,0)
        P6 = ax.c2p(-5,0,0)
        P7 = ax.c2p(-15,-5,0)
        P8 = ax.c2p(-5,-10,0)
        P9 = ax.c2p(0,0,0)
        P10 = ax.c2p(2.5,5,0)
        P11 = ax.c2p(25,5,0)
        P12 = ax.c2p(15,8,0)

        P = np.array([P0,P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12])

        for i in range(13):
            self.add(Circle(radius=0.125/(1 if i%3==0 else 2), color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=5/(1 if i%3==0 else 2)).move_to(P[i]).set_z_index(3))
        
        self.add(Line(
            P0,
            P1,
            stroke_width=3.75
        ).set_z_index(2).set_opacity(0.5))
        self.add(Line(
            P2,
            P3,
            stroke_width=3.75
        ).set_z_index(2).set_opacity(0.5))
        self.add(Line(
            P3,
            P4,
            stroke_width=3.75
        ).set_z_index(2).set_opacity(0.5))
        self.add(Line(
            P5,
            P6,
            stroke_width=3.75
        ).set_z_index(2).set_opacity(0.5))
        self.add(Line(
            P6,
            P7,
            stroke_width=3.75
        ).set_z_index(2).set_opacity(0.5))
        self.add(Line(
            P8,
            P9,
            stroke_width=3.75
        ).set_z_index(2).set_opacity(0.5))
        self.add(Line(
            P9,
            P10,
            stroke_width=3.75
        ).set_z_index(2).set_opacity(0.5))
        self.add(Line(
            P11,
            P12,
            stroke_width=3.75
        ).set_z_index(2).set_opacity(0.5))

        curve0 = ParametricFunction(
            lambda t: self.Bezier(
                t,
                [
                    P0,
                    P1,
                    P2,
                    P3
                ]
            ),
            t_range = (0, 1),
            color = WHITE,
            stroke_width=7.5
        )
        curve1 = ParametricFunction(
            lambda t: self.Bezier(
                t,
                [
                    P3,
                    P4,
                    P5,
                    P6
                ]
            ),
            t_range = (0, 1),
            color = WHITE,
            stroke_width=7.5
        )
        curve2 = ParametricFunction(
            lambda t: self.Bezier(
                t,
                [
                    P6,
                    P7,
                    P8,
                    P9
                ]
            ),
            t_range = (0, 1),
            color = WHITE,
            stroke_width=7.5
        )
        curve3 = ParametricFunction(
            lambda t: self.Bezier(
                t,
                [
                    P9,
                    P10,
                    P11,
                    P12
                ]
            ),
            t_range = (0, 1),
            color = WHITE,
            stroke_width=7.5
        )
        self.add(curve0, curve1, curve2, curve3)


        
        t = Variable(var=0, label='', num_decimal_places=2).move_to([0,20,0])

        r = always_redraw(lambda : Circle(
            radius=0.075, color=BLACK, fill_opacity=1, stroke_color=GREEN, stroke_width=3.5
        ).set_z_index(6).move_to(
            self.Bezier(t.value.get_value()%1,[
                    P[floor(t.value.get_value())*3],
                    P[floor(t.value.get_value())*3+1],
                    P[floor(t.value.get_value())*3+2],
                    P[floor(t.value.get_value())*3+3]
                ])
            if t.value.get_value()<4 else P[12]
        ))
        
        
        rPrime = always_redraw(lambda : Arrow(
            r.get_center(),
            r.get_center() + 1/4*(self.BezierPrime(t.value.get_value()%1,[
                    P[floor(t.value.get_value())*3],
                    P[floor(t.value.get_value())*3+1],
                    P[floor(t.value.get_value())*3+2],
                    P[floor(t.value.get_value())*3+3]
                ]) if t.value.get_value()<4 else self.BezierPrime(1,[
                    P[9],
                    P[10],
                    P[11],
                    P[12]
                ])),
            buff=0,
            color=GREEN,
            stroke_width=7.5
        ).set_z_index(4))

        rT = always_redraw(lambda : Arrow(
            r.get_center(),
            r.get_center() + normalize(self.BezierPrime(t.value.get_value()%1,[
                    P[floor(t.value.get_value())*3],
                    P[floor(t.value.get_value())*3+1],
                    P[floor(t.value.get_value())*3+2],
                    P[floor(t.value.get_value())*3+3]
                ]) if t.value.get_value()<4 else self.BezierPrime(1,[
                    P[9],
                    P[10],
                    P[11],
                    P[12]
                ])),
            buff=0,
            color=BLUE,
            stroke_width=7.5
        ).set_z_index(5).set_opacity(0.95))
        

        self.add(r, rPrime, rT, t)

        self.play(t.tracker.animate.set_value(0), run_time=0.3)
        self.play(t.tracker.animate.set_value(4), run_time=6, rate_func=linear)
        self.play(t.tracker.animate.set_value(4), run_time=0.1)
        
class Slide7_9(Scene):
    def Bezier(self, t, p):
        p0, p1, p2, p3 = p
        return (1-t)**3 * p0 + 3*(1-t)**2 * t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3

    def BezierPrime(self, t, p):
        p0, p1, p2, p3 = p
        return 3*(1-t)**2*(p1 - p0) + 6*(1-t)*t*(p2 - p1) + 3*t**2*(p3 - p2)
    
    def BezierPPrime(self, t, p):
        p0, p1, p2, p3 = p
        return 6 * (1 - t) * (p2 - 2 * p1 + p0) + 6 * t * (p3 - 2 * p2 + p1)
    
    def construct(self):
        self.camera.background_color = '#191919'
        ax = NumberPlane(
            x_range=[-24, 24, 1],
            y_range=[-12, 12, 1],
            x_length=12,
            y_length=6,
        )

        P0 = ax.c2p(-25,-10,0)
        P1 = ax.c2p(-24,-3,0)
        P2 = ax.c2p(-20,2,0)
        P3 = ax.c2p(-23,5,0)
        P4 = ax.c2p(-10,10,0)
        P5 = ax.c2p(5,5,0)
        P6 = ax.c2p(-5,0,0)
        P7 = ax.c2p(-15,-5,0)
        P8 = ax.c2p(-5,-10,0)
        P9 = ax.c2p(5,0,0)
        P10 = ax.c2p(10,5,0)
        P11 = ax.c2p(8,10,0)
        P12 = ax.c2p(15,8,0)

        P = np.array([P0,P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12])

        for i in range(13):
            self.add(Circle(radius=0.125/(1 if i%3==0 else 2), color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=5/(1 if i%3==0 else 2)).move_to(P[i]).set_z_index(3))
        
        self.add(Line(
            P0,
            P1,
            stroke_width=3.75
        ).set_z_index(2).set_opacity(0.5))
        self.add(Line(
            P2,
            P3,
            stroke_width=3.75
        ).set_z_index(2).set_opacity(0.5))
        self.add(Line(
            P3,
            P4,
            stroke_width=3.75
        ).set_z_index(2).set_opacity(0.5))
        self.add(Line(
            P5,
            P6,
            stroke_width=3.75
        ).set_z_index(2).set_opacity(0.5))
        self.add(Line(
            P6,
            P7,
            stroke_width=3.75
        ).set_z_index(2).set_opacity(0.5))
        self.add(Line(
            P8,
            P9,
            stroke_width=3.75
        ).set_z_index(2).set_opacity(0.5))
        self.add(Line(
            P9,
            P10,
            stroke_width=3.75
        ).set_z_index(2).set_opacity(0.5))
        self.add(Line(
            P11,
            P12,
            stroke_width=3.75
        ).set_z_index(2).set_opacity(0.5))

        curve0 = ParametricFunction(
            lambda t: self.Bezier(
                t,
                [
                    P0,
                    P1,
                    P2,
                    P3
                ]
            ),
            t_range = (0, 1),
            color = WHITE,
            stroke_width=7.5
        )
        curve1 = ParametricFunction(
            lambda t: self.Bezier(
                t,
                [
                    P3,
                    P4,
                    P5,
                    P6
                ]
            ),
            t_range = (0, 1),
            color = WHITE,
            stroke_width=7.5
        )
        curve2 = ParametricFunction(
            lambda t: self.Bezier(
                t,
                [
                    P6,
                    P7,
                    P8,
                    P9
                ]
            ),
            t_range = (0, 1),
            color = WHITE,
            stroke_width=7.5
        )
        curve3 = ParametricFunction(
            lambda t: self.Bezier(
                t,
                [
                    P9,
                    P10,
                    P11,
                    P12
                ]
            ),
            t_range = (0, 1),
            color = WHITE,
            stroke_width=7.5
        )
        self.add(curve0, curve1, curve2, curve3)


        
        t = Variable(var=0, label='', num_decimal_places=2).move_to([0,20,0])

        r = always_redraw(lambda : Circle(
            radius=0.075, color=BLACK, fill_opacity=1, stroke_color=GREEN, stroke_width=3.5
        ).set_z_index(6).move_to(
            self.Bezier(t.value.get_value()%1,[
                    P[floor(t.value.get_value())*3],
                    P[floor(t.value.get_value())*3+1],
                    P[floor(t.value.get_value())*3+2],
                    P[floor(t.value.get_value())*3+3]
                ])
            if t.value.get_value()<4 else P[12]
        ))
        
        
        rPrime = always_redraw(lambda : Arrow(
            r.get_center(),
            r.get_center() + 1/4*(self.BezierPrime(t.value.get_value()%1,[
                    P[floor(t.value.get_value())*3],
                    P[floor(t.value.get_value())*3+1],
                    P[floor(t.value.get_value())*3+2],
                    P[floor(t.value.get_value())*3+3]
                ]) if t.value.get_value()<4 else self.BezierPrime(1,[
                    P[9],
                    P[10],
                    P[11],
                    P[12]
                ])),
            buff=0,
            color=GREEN,
            stroke_width=7.5
        ).set_z_index(4))


        rPPrime = always_redraw(lambda : Arrow(
            rPrime.get_end(),
            rPrime.get_end() + 1/15*(self.BezierPPrime(t.value.get_value()%1,[
                    P[floor(t.value.get_value())*3],
                    P[floor(t.value.get_value())*3+1],
                    P[floor(t.value.get_value())*3+2],
                    P[floor(t.value.get_value())*3+3]
                ]) if t.value.get_value()<4 else self.BezierPPrime(1,[
                    P[9],
                    P[10],
                    P[11],
                    P[12]
                ])),
            buff=0,
            color=RED,
            stroke_width=7.5
        ).set_z_index(3))


        n_spikes = 160
        n_comb_points = 8000

        t_samples_spikes = np.linspace(0, 4, n_spikes)
        t_samples_comb = np.linspace(0, 4, n_comb_points)

        spikes = VGroup()
        comb_curves = VGroup()  # Store separate comb pieces

        # Create spikes (all at once, over full t range)
        for t_val in t_samples_spikes:
            if t_val < 4:
                i = floor(t_val)
                local_t = t_val % 1
                rP = self.BezierPrime(local_t, [P[i*3], P[i*3+1], P[i*3+2], P[i*3+3]])
                rPP = self.BezierPPrime(local_t, [P[i*3], P[i*3+1], P[i*3+2], P[i*3+3]])
                r_pos = self.Bezier(local_t, [P[i*3], P[i*3+1], P[i*3+2], P[i*3+3]])
            else:
                rP = self.BezierPrime(1, [P[9], P[10], P[11], P[12]])
                rPP = self.BezierPPrime(1, [P[9], P[10], P[11], P[12]])
                r_pos = P[12]

            tangent = rP / np.linalg.norm(rP)
            curvature = np.linalg.norm(np.cross(rP, rPP)) / (np.linalg.norm(rP)**3)

            sign = np.cross(rP, rPP)[2]
            normal = normalize(np.array([-rP[1], rP[0], 0]) if sign < 0 else np.array([rP[1], -rP[0], 0]))

            spike_length = 0.7 * curvature
            spike = Line(
                r_pos,
                r_pos + spike_length * normal,
                buff=0,
                stroke_width=2,
                color=WHITE,
            )
            spikes.add(spike)

        # Now create comb curves separately between knots
        for i in range(4):  # 5 spline segments between P0-P1, P1-P2, ..., P4-P5
            # Only sample t in [i, i+1]
            t_subsamples = np.linspace(i, i+1, n_comb_points//4)

            tip_points = []

            for t_val in t_subsamples:
                local_t = t_val - i  # always between 0 and 1
                rP = self.BezierPrime(local_t, [P[i*3], P[i*3+1], P[i*3+2], P[i*3+3]])
                rPP = self.BezierPPrime(local_t, [P[i*3], P[i*3+1], P[i*3+2], P[i*3+3]])
                r_pos = self.Bezier(local_t, [P[i*3], P[i*3+1], P[i*3+2], P[i*3+3]])

                tangent = rP / np.linalg.norm(rP)
                curvature = np.linalg.norm(np.cross(rP, rPP)) / (np.linalg.norm(rP)**3)

                sign = np.cross(rP, rPP)[2]
                normal = normalize(np.array([-rP[1], rP[0], 0]) if sign < 0 else np.array([rP[1], -rP[0], 0]))

                spike_length = 0.7 * curvature
                tip_point = r_pos + spike_length * normal
                tip_points.append(tip_point)

            # Make a separate comb curve for each segment
            comb_curve = VMobject(color=WHITE, stroke_width=2)
            comb_curve.set_points_as_corners(tip_points)
            comb_curves.add(comb_curve)
        

        self.add(r, rPrime, rPPrime, t)

        self.play(t.tracker.animate.set_value(0), run_time=0.3)
        self.play(Create(spikes), Create(comb_curves), t.tracker.animate.set_value(4), run_time=6, rate_func=linear)
        self.play(t.tracker.animate.set_value(4), run_time=0.1)
       
class Slide8_2(Scene):
    def construct(self):
        self.camera.background_color = '#191919'
        ax1 = Axes(
            x_range=[0, 1, 0.2],
            y_range=[0, 1, 0.2],
            x_length=3,
            y_length=3,
            tips=False,
            axis_config={"include_numbers": True, "font_size": 18}
        ).set_z_index(2)
        xLabel = MathTex("t").move_to([1.5+0.25,-1.5,0])
        yLabel = MathTex("w").move_to([-1.5,1.5+0.25,0])

        self.add(ax1, xLabel, yLabel)

        f1 = ParametricFunction(
            lambda t : ax1.coords_to_point(t,1/4*t**3),
            t_range = (0, 1),
            color = RED,
        )
        f2 = ParametricFunction(
            lambda t : ax1.coords_to_point(t,2/3*(1/2*(t)-1)**3+11/12-1/4*(t)),
            t_range = (0, 1),
            color = GREEN,
        )
        f3 = ParametricFunction(
            lambda t : ax1.coords_to_point(t,2/3*(1/2*(1-t)-1)**3+11/12-1/4*(1-t)),
            t_range = (0, 1),
            color = BLUE,
        )
        f4 = ParametricFunction(
            lambda t : ax1.coords_to_point(t,1/4*(1-t)**3),
            t_range = (0, 1),
            color = YELLOW,
        )

        self.add(f1,f2,f3,f4)



        ax2 = Axes(
            x_range=[0, 4, 0.2],
            y_range=[0, 1, 0.2],
            x_length=12,
            y_length=3,
            tips=False,
            axis_config={"include_numbers": True, "font_size": 18}
        ).set_z_index(2)
        xLabel2 = MathTex("t").move_to([6+0.25,-1.5,0])
        yLabel2 = MathTex("w").move_to([-6,1.5+0.25,0])

        f12 = ParametricFunction(
            lambda t : ax2.coords_to_point(t,1/4*t**3),
            t_range = (0, 1),
            color = RED,
        )
        f22 = ParametricFunction(
            lambda t : ax2.coords_to_point(t,2/3*(1/2*(t-1)-1)**3+11/12-1/4*(t-1)),
            t_range = (1, 2),
            color = GREEN,
        )
        f32 = ParametricFunction(
            lambda t : ax2.coords_to_point(t,2/3*(1/2*(3-t)-1)**3+11/12-1/4*(3-t)),
            t_range = (2, 3),
            color = BLUE,
        )
        f42 = ParametricFunction(
            lambda t : ax2.coords_to_point(t,1/4*(4-t)**3),
            t_range = (3, 4),
            color = YELLOW,
        )

        self.wait(1)
        self.play(
            ReplacementTransform(ax1,ax2),
            ReplacementTransform(xLabel,xLabel2),
            ReplacementTransform(yLabel,yLabel2),
            ReplacementTransform(f1,f12),
            ReplacementTransform(f2,f22),
            ReplacementTransform(f3,f32),
            ReplacementTransform(f4,f42)
        )

class Slide8_6(Scene):
    def construct(self):
        self.camera.background_color = '#191919'
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

        f1 = ParametricFunction(
            lambda t : ax.coords_to_point(t,1/6*(1-3*t+3*t**2-t**3)),
            t_range = (0, 1),
            color = RED,
        )
        f2 = ParametricFunction(
            lambda t : ax.coords_to_point(t,1/6*(4-6*t**2+3*t**3)),
            t_range = (0, 1),
            color = GREEN,
        )
        f3 = ParametricFunction(
            lambda t : ax.coords_to_point(t,1/6*(1+3*t+3*t**2-3*t**3)),
            t_range = (0, 1),
            color = BLUE,
        )
        f4 = ParametricFunction(
            lambda t : ax.coords_to_point(t,1/6*t**3),
            t_range = (0, 1),
            color = YELLOW,
        )

        self.play(
            Create(f1),
            Create(f2),
            Create(f3),
            Create(f4),
            run_time=2
        )

class Slide8_7(Scene):
    def BSpline(self, t, p):
        p0, p1, p2, p3 = p
        B0 = (1 - t)**3 / 6
        B1 = (3*t**3 - 6*t**2 + 4) / 6
        B2 = (-3*t**3 + 3*t**2 + 3*t + 1) / 6
        B3 = t**3 / 6
        return B0 * p0 + B1 * p1 + B2 * p2 + B3 * p3

    def BSplinePrime(self, t, p):
        p0, p1, p2, p3 = p
        B0_prime = -0.5 * (1 - t)**2
        B1_prime = (1.5 * t**2 - 2 * t)
        B2_prime = (-1.5 * t**2 + t + 0.5)
        B3_prime = 0.5 * t**2
        return B0_prime * p0 + B1_prime * p1 + B2_prime * p2 + B3_prime * p3

    def BSplinePPrime(self, t, p):
        p0, p1, p2, p3 = p
        B0_pprime = (1 - t)
        B1_pprime = (3 * t - 2)
        B2_pprime = (-3 * t + 1)
        B3_pprime = t
        return B0_pprime * p0 + B1_pprime * p1 + B2_pprime * p2 + B3_pprime * p3

    def construct(self):
        self.camera.background_color = '#191919'
        ax = NumberPlane(
            x_range=[-24, 24, 1],
            y_range=[-12, 12, 1],
            x_length=12,
            y_length=6,
        )

        P0 = ax.c2p(-15,-12,0)
        P1 = ax.c2p(-25,-5,0)
        P2 = ax.c2p(-10,10,0)
        P3 = ax.c2p(25,5,0)
        P4 = ax.c2p(18,-10,0)
        P5 = ax.c2p(5,-7,0)
        P6 = ax.c2p(-10,-12,0)
        P7 = ax.c2p(-12,3,0)
        P8 = ax.c2p(5,-3,0)
        P9 = ax.c2p(2,5,0)

        P = np.array([P0,P1,P2,P3,P4,P5,P6,P7,P8,P9])

        for e in P:
            self.add(Circle(radius=0.125, color=BLACK, fill_opacity=1, stroke_color=RED, stroke_width=5).move_to(e).set_z_index(3))
        
        for i in range(9):
            self.add(Line(
            P[i],
            P[i+1],
            stroke_width=3.75
        ).set_z_index(2).set_opacity(0.5))

        for i in range(7):
            self.add(ParametricFunction(
                lambda t: self.BSpline(
                    t,
                    [
                        P[i],
                        P[i+1],
                        P[i+2],
                        P[i+3]
                    ]
                ),
                t_range = (0, 1),
                color = WHITE,
                stroke_width=7.5
            ),
            Circle(radius=0.0625, color=BLACK, fill_opacity=1, stroke_color=WHITE, stroke_width=2.5).move_to(self.BSpline(
                    0,
                    [
                        P[i],
                        P[i+1],
                        P[i+2],
                        P[i+3]
                    ])
            ).set_z_index(3))
        self.add(Circle(radius=0.0625, color=BLACK, fill_opacity=1, stroke_color=WHITE, stroke_width=2.5).move_to(self.BSpline(
                    1,
                    [
                        P[6],
                        P[7],
                        P[8],
                        P[9]
                    ])
            ).set_z_index(3))



        t = Variable(var=0, label='', num_decimal_places=2).move_to([0,20,0])

        r = always_redraw(lambda : Circle(
            radius=0.075, color=BLACK, fill_opacity=1, stroke_color=GREEN, stroke_width=3.5
        ).set_z_index(5).move_to(
            self.BSpline(t.value.get_value()%1,[
                    P[floor(t.value.get_value())],
                    P[floor(t.value.get_value())+1],
                    P[floor(t.value.get_value())+2],
                    P[floor(t.value.get_value())+3]
                ])
            if t.value.get_value()<7 else self.BSpline(
                    1,
                    [
                        P[6],
                        P[7],
                        P[8],
                        P[9]
                    ])
        ))
        
        
        rPrime = always_redraw(lambda : Arrow(
            r.get_center(),
            r.get_center() + 1/3*(self.BSplinePrime(t.value.get_value()%1,[
                    P[floor(t.value.get_value())],
                    P[floor(t.value.get_value())+1],
                    P[floor(t.value.get_value())+2],
                    P[floor(t.value.get_value())+3]
                ]) if t.value.get_value()<7 else self.BSplinePrime(
                    1,
                    [
                        P[6],
                        P[7],
                        P[8],
                        P[9]
                    ])),
            buff=0,
            color=GREEN,
            stroke_width=7.5
        ).set_z_index(4))

        rPPrime = always_redraw(lambda : Arrow(
            rPrime.get_end(),
            rPrime.get_end() + 1/5*(self.BSplinePPrime(t.value.get_value()%1,[
                    P[floor(t.value.get_value())],
                    P[floor(t.value.get_value())+1],
                    P[floor(t.value.get_value())+2],
                    P[floor(t.value.get_value())+3]
                ]) if t.value.get_value()<7 else self.BSplinePPrime(1,[
                    P[6],
                    P[7],
                    P[8],
                    P[9]
                ])),
            buff=0,
            color=RED,
            stroke_width=7.5
        ).set_z_index(3))



        n_spikes = 280
        n_comb_points = 14000

        t_samples_spikes = np.linspace(0, 7, n_spikes)
        t_samples_comb = np.linspace(0, 7, n_comb_points)

        spikes = VGroup()
        comb_curves = VGroup()  # Store separate comb pieces

        # Create spikes (all at once, over full t range)
        for t_val in t_samples_spikes:
            if t_val < 7:
                i = floor(t_val)
                local_t = t_val % 1
                rP = self.BSplinePrime(local_t, [P[i], P[i+1], P[i+2], P[i+3]])
                rPP = self.BSplinePPrime(local_t, [P[i], P[i+1], P[i+2], P[i+3]])
                r_pos = self.BSpline(local_t, [P[i], P[i+1], P[i+2], P[i+3]])
            else:
                rP = self.BSplinePrime(local_t, [P[i], P[i+1], P[i+2], P[i+3]])
                rPP = self.BSplinePPrime(local_t, [P[i], P[i+1], P[i+2], P[i+3]])
                r_pos = self.BSpline(1, [P[6], P[7], P[8], P[9]])

            tangent = rP / np.linalg.norm(rP)
            curvature = np.linalg.norm(np.cross(rP, rPP)) / (np.linalg.norm(rP)**3)

            sign = np.cross(rP, rPP)[2]
            normal = normalize(np.array([-rP[1], rP[0], 0]) if sign < 0 else np.array([rP[1], -rP[0], 0]))

            spike_length = 1 * curvature
            spike = Line(
                r_pos,
                r_pos + spike_length * normal,
                buff=0,
                stroke_width=2,
                color=WHITE,
            )
            spikes.add(spike)

        # Now create comb curves separately between knots
        for i in range(7):  # 5 spline segments between P0-P1, P1-P2, ..., P4-P5
            # Only sample t in [i, i+1]
            t_subsamples = np.linspace(i, i+1, n_comb_points//7)

            tip_points = []

            for t_val in t_subsamples:
                local_t = t_val - i  # always between 0 and 1
                rP = self.BSplinePrime(local_t, [P[i], P[i+1], P[i+2], P[i+3]])
                rPP = self.BSplinePPrime(local_t, [P[i], P[i+1], P[i+2], P[i+3]])
                r_pos = self.BSpline(local_t, [P[i], P[i+1], P[i+2], P[i+3]])

                tangent = rP / np.linalg.norm(rP)
                curvature = np.linalg.norm(np.cross(rP, rPP)) / (np.linalg.norm(rP)**3)

                sign = np.cross(rP, rPP)[2]
                normal = normalize(np.array([-rP[1], rP[0], 0]) if sign < 0 else np.array([rP[1], -rP[0], 0]))

                spike_length = 1 * curvature
                tip_point = r_pos + spike_length * normal
                tip_points.append(tip_point)

            # Make a separate comb curve for each segment
            comb_curve = VMobject(color=WHITE, stroke_width=2)
            comb_curve.set_points_as_corners(tip_points)
            comb_curves.add(comb_curve)


        

        self.add(r, rPrime, rPPrime, t)

        self.play(t.tracker.animate.set_value(0), run_time=0.3)
        self.play(Create(spikes), Create(comb_curves), t.tracker.animate.set_value(7), run_time=7, rate_func=linear)
        self.play(t.tracker.animate.set_value(7), run_time=0.1)

class Slide9_2(ThreeDScene):
    def construct(self):
        self.camera.background_color = '#191919'
        #self.camera.light_source.move_to(OUT*20)
        self.set_camera_orientation(phi=2*PI/5, theta=PI/4, frame_center=2*OUT, zoom=1)
        box = Cube(side_length=4, fill_opacity=0, stroke_opacity=1, stroke_width=1).shift(2*OUT)
        R = Arrow3D(start=ORIGIN, end=4*RIGHT, thickness=0.02, height=0.3, base_radius=0.08, color=RED, resolution=24).shift(2*(LEFT+DOWN))
        G = Arrow3D(start=ORIGIN, end=4*UP, thickness=0.02, height=0.3, base_radius=0.08, color=GREEN, resolution=24).shift(2*(LEFT+DOWN))
        B = Arrow3D(start=ORIGIN, end=4*OUT, thickness=0.02, height=0.3, base_radius=0.08, color=BLUE, resolution=24).shift(2*(LEFT+DOWN))

        self.add(box, R,G,B)

        # Control points for the Bezier curve (choose 4 arbitrary points inside the cube)
        p0 = np.array([0, -2, 0])
        p1 = np.array([2, 2, 1])
        p2 = np.array([-2, 2, 2])
        p3 = np.array([-2, -2, 3])

        # Function to compute cubic Bezier
        def bezier(t):
            return (
                (1 - t)**3 * p0 +
                3 * (1 - t)**2 * t * p1 +
                3 * (1 - t) * t**2 * p2 +
                t**3 * p3
            )

        # Now sample points along the Bezier curve
        n_samples = 200
        t_values = np.linspace(0, 1, n_samples)
        bezier_points = [bezier(t) for t in t_values]

        # Create small line segments between consecutive points
        lines = VGroup()
        for i in range(len(bezier_points)-1):
            start = bezier_points[i]
            end = bezier_points[i+1]
            # Map position to RGB color
            r = np.clip((start[0]+2)/4, 0, 1)
            g = np.clip((start[1]+2)/4, 0, 1)
            b = np.clip((start[2]-2)/4, 0, 1)
            color = rgb_to_color([r, g, b])

            line = Line3D(
                start=start,
                end=end,
                thickness=0.02,
                color=color,
            )
            lines.add(line)

        self.add(lines)

        self.begin_ambient_camera_rotation(PI/4)
        self.wait(8)

class Slide9_3(Scene):
    def construct(self):
        # Control points for the Bezier curve (choose 4 arbitrary points inside the cube)
        p0 = np.array([0, -2, 0])
        p1 = np.array([2, 2, 1])
        p2 = np.array([-2, 2, 2])
        p3 = np.array([-2, -2, 3])

        # Function to compute cubic Bezier
        def bezier(t):
            return (
                (1 - t)**3 * p0 +
                3 * (1 - t)**2 * t * p1 +
                3 * (1 - t) * t**2 * p2 +
                t**3 * p3
            )

        # Now sample points along the Bezier curve
        n_samples = 1000
        t_values = np.linspace(0, 1, n_samples)
        bezier_points = [bezier(t) for t in t_values]

        # Create small line segments between consecutive points
        lines = VGroup()
        for i in range(len(bezier_points)-1):
            start = [-6+12*i/n_samples,0,0]
            end = [-6+12*(i+1)/n_samples,0,0]
            # Map position to RGB color
            r = np.clip((bezier_points[i][0]+2)/4, 0, 1)
            g = np.clip((bezier_points[i][1]+2)/4, 0, 1)
            b = np.clip((bezier_points[i][2]-2)/4, 0, 1)
            color = rgb_to_color([r, g, b])

            line = Line3D(
                start=start,
                end=end,
                thickness=1,
                color=color,
            )
            lines.add(line)

        self.add(lines)

        self.add(
            Rectangle(color=ManimColor(BLACK), width=12,height=2, stroke_width=6),
            Rectangle(color=ManimColor(WHITE), width=12,height=2, stroke_width=2)
        )

