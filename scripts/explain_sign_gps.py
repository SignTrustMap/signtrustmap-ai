from manim import *
import numpy as np
import math

BG_COLOR = "#0b0f19"
CAM      = "#58a6ff"
SIGN     = "#ff7b72"
RAY      = "#f1e05a"
FORM     = "#7ee787"
ROAD     = "#21262d"
WHITE_C  = "#e6edf3"
GREY     = "#8b949e"
DIM      = "#161b22"
config.media_dir = "outputs"
config.background_color = BG_COLOR
config.max_files_cached = 1000


class SignGPSExplainer(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        title = MathTex(
            r"\text{Sign GPS Estimation: Multi-View Geometry } \& \text{ Geodetic Triangulation}",
            font_size=26, color=WHITE_C
        ).to_edge(UP, buff=0.45)
        self.play(FadeIn(title, run_time=0.4))

        road = Rectangle(width=12, height=1.6, fill_color=ROAD, fill_opacity=0.9, stroke_width=0).shift(DOWN * 1.5)
        lane = DashedLine(LEFT * 6 + DOWN * 1.5, RIGHT * 6 + DOWN * 1.5, color=GREY, stroke_width=2, dash_length=0.35)

        veh_body = RoundedRectangle(corner_radius=0.1, width=1.1, height=0.55, fill_color=CAM, fill_opacity=0.85, stroke_color=WHITE_C, stroke_width=1.5).shift(LEFT * 4.2 + DOWN * 1.85)
        cam_lens = Dot(veh_body.get_right(), color=RAY, radius=0.07)
        veh = VGroup(veh_body, cam_lens)
        veh_lbl = MathTex(r"\text{Vehicle } (\mathbf{c}_i, \psi_i)", font_size=18, color=CAM).next_to(veh, DOWN, buff=0.12)

        sign_post = Line(DOWN * 2.3, UP * 1.3, color=GREY, stroke_width=3).shift(RIGHT * 4.2)
        sign_board = RegularPolygon(n=8, radius=0.36, fill_color=SIGN, fill_opacity=0.9, stroke_color=WHITE_C, stroke_width=1.5).move_to(sign_post.get_top())
        sign_txt = Text("STOP", font_size=9, weight=BOLD, color=WHITE_C).move_to(sign_board)
        sign_obj = VGroup(sign_post, sign_board, sign_txt)
        sign_lbl = MathTex(r"\mathbf{s}^* = (\phi_s, \lambda_s) \ ?", font_size=20, color=SIGN).next_to(sign_board, UP, buff=0.15)

        sight_ray_1 = Arrow(cam_lens.get_center(), sign_board.get_left(), color=RAY, stroke_width=2.5, buff=0.08, tip_length=0.18)
        sight_ray_lbl_1 = MathTex(r"\text{Sight Ray}", font_size=16, color=RAY).next_to(sight_ray_1.get_center(), UP, buff=0.1).shift(LEFT * 0.4)

        photo_frame = RoundedRectangle(corner_radius=0.08, width=3.2, height=2.0, fill_color="#0e131f", fill_opacity=0.95, stroke_color=GREY, stroke_width=1.5).move_to(LEFT * 2.5 + UP * 1.3)
        photo_title = MathTex(r"\text{Camera View } \mathbf{I}_i", font_size=17, color=GREY).next_to(photo_frame, UP, buff=0.08)
        
        pt_A_photo = photo_frame.get_center()
        dot_A_photo = Dot(pt_A_photo, color=WHITE_C, radius=0.06)
        lbl_A_photo = MathTex(r"A", font_size=15, color=GREY).next_to(dot_A_photo, DOWN, buff=0.06)

        pt_B_photo = pt_A_photo + RIGHT * 0.75 + UP * 0.15
        bbox = Rectangle(width=0.6, height=0.75, color=SIGN, stroke_width=1.8).move_to(pt_B_photo)
        dot_B_photo = Dot(pt_B_photo, color=SIGN, radius=0.06)
        lbl_B_photo = MathTex(r"B", font_size=15, color=SIGN).next_to(bbox, UR, buff=0.06)
        photo_group = VGroup(photo_frame, photo_title, dot_A_photo, lbl_A_photo, bbox, dot_B_photo, lbl_B_photo)

        problem_formula = MathTex(
            r"\text{Given: } (\phi_i, \lambda_i, \psi_i) \text{ and 2D detection } B(u, v) "
            r"\implies \text{Find true 3D GPS } \mathbf{s}^*",
            font_size=23, color=FORM
        ).to_edge(DOWN, buff=0.45)

        self.play(FadeIn(road), Create(lane))
        self.play(FadeIn(veh), Write(veh_lbl))
        self.play(FadeIn(sign_obj), Write(sign_lbl))
        self.play(GrowArrow(sight_ray_1), Write(sight_ray_lbl_1))
        self.play(FadeIn(photo_group, shift=DOWN * 0.2))
        self.play(FadeIn(problem_formula, shift=UP * 0.15))
        self.wait(1.5)

        title_pinhole = MathTex(
            r"\text{Pinhole Optics: } \triangle CAB \implies \tan(\Delta\theta) = \frac{\Delta x}{f}",
            font_size=26, color=WHITE_C
        ).to_edge(UP, buff=0.45)

        f_val  = 2.4
        dx_val = 1.1

        pt_C1 = np.array([-3.8, -1.2, 0])
        pt_A  = pt_C1 + np.array([f_val, 0, 0])
        pt_B  = pt_A + np.array([0, dx_val, 0])
        
        pt_S_star = np.array([1.8, 1.3, 0])
        ray1_dir = pt_S_star - pt_C1
        ray1_unit = ray1_dir / np.linalg.norm(ray1_dir)
        ray1_end = pt_C1 + 1.25 * ray1_dir

        dot_C1 = Dot(pt_C1, color=CAM, radius=0.14)
        lbl_C1 = MathTex(r"C \text{ (Camera)}", font_size=21, color=CAM).next_to(dot_C1, DL, buff=0.12)

        dot_A = Dot(pt_A, color=WHITE_C, radius=0.09)
        lbl_A = MathTex(r"A \text{ (Image Center)}", font_size=19, color=GREY).next_to(dot_A, DR, buff=0.14)

        dot_B = Dot(pt_B, color=SIGN, radius=0.11)
        lbl_B = MathTex(r"B \text{ (Detection)}", font_size=19, color=SIGN).next_to(dot_B, UL, buff=0.12)

        line_CA = Line(pt_C1, pt_A, color=GREY, stroke_width=2.5)
        line_AB = Line(pt_A, pt_B, color=WHITE_C, stroke_width=3)

        image_plane = Line(pt_A + DOWN * 0.8, pt_A + UP * 1.6, color="#384357", stroke_width=1.5)
        image_plane_tag = Text("Image Plane", font_size=11, color="#384357").next_to(image_plane, DOWN, buff=0.12)

        opt_axis = DashedLine(pt_C1, pt_C1 + RIGHT * 7.5, color="#21262d", stroke_width=1.2, dash_length=0.15)
        opt_axis_tag = MathTex(r"\text{Optical Axis (Heading } \psi\text{)}", font_size=15, color=GREY).next_to(opt_axis.get_end(), DOWN, buff=0.1).shift(LEFT * 1.2)

        sq_right = Polygon(
            pt_A, pt_A + LEFT * 0.22, pt_A + LEFT * 0.22 + UP * 0.22, pt_A + UP * 0.22,
            color=GREY, stroke_width=1.5
        )

        ray_arrow = Arrow(pt_C1, pt_C1 + 1.8 * (pt_B - pt_C1), color=RAY, stroke_width=3, buff=0, tip_length=0.22)
        ray_lbl   = MathTex(r"\text{Sight Ray}", font_size=19, color=RAY).next_to(ray_arrow.get_end(), UR, buff=0.1)

        arrow_f = DoubleArrow(pt_C1 + DOWN * 0.35, pt_A + DOWN * 0.35, color=FORM, stroke_width=2, buff=0, tip_length=0.12)
        lbl_f   = MathTex(r"f", font_size=23, color=FORM).next_to(arrow_f, DOWN, buff=0.08)

        arrow_dx = DoubleArrow(pt_A + RIGHT * 0.25, pt_B + RIGHT * 0.25, color=WHITE_C, stroke_width=2, buff=0, tip_length=0.12)
        lbl_dx   = MathTex(r"\Delta x", font_size=23, color=WHITE_C).next_to(arrow_dx, RIGHT, buff=0.08)

        ang = math.atan2(dx_val, f_val)
        arc_dth = Arc(radius=1.2, arc_center=pt_C1, start_angle=0, angle=ang, color=RAY, stroke_width=2.5)
        lbl_dth = MathTex(r"\Delta\theta", font_size=22, color=RAY).move_to(
            pt_C1 + 1.55 * np.array([math.cos(ang / 2), math.sin(ang / 2), 0])
        )

        formula_pinhole = MathTex(
            r"\text{Right } \triangle CAB \ (\angle A = 90^\circ): \quad \tan(\Delta\theta) = \frac{AB}{CA} = \frac{\Delta x}{f} "
            r"\implies \Delta\theta = \arctan\!\left(\frac{\Delta x}{f}\right)",
            font_size=24, color=FORM
        ).to_edge(DOWN, buff=0.45)

        self.play(
            Transform(title, title_pinhole),
            Transform(problem_formula, formula_pinhole),
            FadeOut(VGroup(road, lane, veh, veh_lbl, sign_obj, sign_lbl, photo_group)),
            ReplacementTransform(sight_ray_1, ray_arrow),
            ReplacementTransform(sight_ray_lbl_1, ray_lbl),
            FadeIn(dot_C1),
            Write(lbl_C1),
            run_time=0.7
        )
        self.play(Create(opt_axis), Write(opt_axis_tag), run_time=0.6)
        self.play(Create(image_plane), Write(image_plane_tag), run_time=0.6)
        self.play(FadeIn(dot_A), Write(lbl_A), Create(line_CA), Create(sq_right), run_time=0.6)
        self.play(Create(arrow_f), Write(lbl_f), run_time=0.5)
        self.play(FadeIn(dot_B), Write(lbl_B), Create(line_AB), run_time=0.6)
        self.play(Create(arrow_dx), Write(lbl_dx), run_time=0.5)
        self.play(Create(arc_dth), Write(lbl_dth), run_time=0.6)
        self.wait(1.5)

        title_global = MathTex(
            r"\text{Absolute Bearing: } \theta_1 = \psi_1 + \Delta\theta_1 \implies \text{Sign Lies on 1D Ray}",
            font_size=26, color=WHITE_C
        ).to_edge(UP, buff=0.45)

        lbl_c1_gps = MathTex(r"\mathbf{c}_1", font_size=21, color=CAM).next_to(dot_C1, DOWN, buff=0.15)
        
        north_ref1 = DashedLine(pt_C1, pt_C1 + UP * 2.8, color=GREY, stroke_width=2, dash_length=0.12)
        north_lbl1 = MathTex(r"\mathbf{n} \text{ (North)}", font_size=16, color=GREY).next_to(north_ref1.get_end(), UP, buff=0.08)

        total_arc1 = Arc(radius=1.3, arc_center=pt_C1, start_angle=PI / 2, angle=-(PI / 2 - ang), color=RAY, stroke_width=2.5)
        mid_bearing = (PI / 2 + ang) / 2
        th_pos1 = pt_C1 + 1.7 * np.array([math.cos(mid_bearing), math.sin(mid_bearing), 0])
        th_tex1 = MathTex(r"\theta_1", font_size=20, color=RAY).next_to(th_pos1, UR, buff=0.08)

        ray1_arrow_full = Arrow(pt_C1, ray1_end, color=RAY, stroke_width=3, buff=0, tip_length=0.22)
        ray1_lbl_full = MathTex(r"\mathbf{u}_1", font_size=19, color=RAY).next_to(ray1_end, UR, buff=0.08)

        d_tracker = ValueTracker(2.4)
        slide_dot = always_redraw(lambda: Dot(pt_C1 + d_tracker.get_value() * ray1_unit, color=SIGN, radius=0.15))
        slide_lbl = always_redraw(lambda: MathTex(r"\mathbf{s}(d)", font_size=20, color=SIGN).next_to(slide_dot, DR, buff=0.12))

        formula_ray = MathTex(
            r"\mathbf{s}(d) = \mathbf{c}_1 + d \cdot \mathbf{u}(\theta_1) \quad (\text{Distance } d > 0 \text{ is unknown with 1 view})",
            font_size=23, color=SIGN
        ).to_edge(DOWN, buff=0.45)

        internal_items = VGroup(
            dot_A, lbl_A, dot_B, lbl_B, line_CA, line_AB, image_plane, image_plane_tag,
            sq_right, arrow_f, lbl_f, arrow_dx, lbl_dx, arc_dth, lbl_dth, ray_lbl
        )

        self.play(
            Transform(title, title_global),
            Transform(problem_formula, formula_ray),
            Transform(lbl_C1, lbl_c1_gps),
            FadeOut(internal_items),
            Create(north_ref1),
            Write(north_lbl1),
            Transform(ray_arrow, ray1_arrow_full),
            Write(ray1_lbl_full),
            run_time=0.8
        )
        self.play(Create(total_arc1), Write(th_tex1), run_time=0.6)
        self.play(FadeIn(slide_dot), FadeIn(slide_lbl))
        self.play(d_tracker.animate.set_value(5.4), run_time=1.0, rate_func=smooth)
        self.play(d_tracker.animate.set_value(2.2), run_time=0.8, rate_func=smooth)
        self.play(d_tracker.animate.set_value(4.2), run_time=0.6, rate_func=smooth)
        self.wait(1.0)

        title_multi = MathTex(
            r"\text{Multi-View Triangulation: Successive Rays Intersect at Unique } \mathbf{s}^*",
            font_size=26, color=WHITE_C
        ).to_edge(UP, buff=0.45)

        formula_intersect = MathTex(
            r"\text{Each view gives ray } i: \ \mathbf{n}_i^T (\mathbf{s} - \mathbf{c}_i) = 0 "
            r"\implies \mathbf{s}^* \text{ is the exact intersection point}",
            font_size=23, color=FORM
        ).to_edge(DOWN, buff=0.45)

        pt_C2 = np.array([-1.2, -1.2, 0])
        pt_C3 = np.array([1.0, -1.2, 0])

        dot_C2 = Dot(pt_C2, color=CAM, radius=0.14)
        lbl_C2 = MathTex(r"\mathbf{c}_2", font_size=21, color=CAM).next_to(dot_C2, DOWN, buff=0.15)
        
        dot_C3 = Dot(pt_C3, color=CAM, radius=0.14)
        lbl_C3 = MathTex(r"\mathbf{c}_3", font_size=21, color=CAM).next_to(dot_C3, DOWN, buff=0.15)

        road_path = DashedLine(pt_C1 + LEFT * 1.5, pt_C3 + RIGHT * 2.5, color="#30363d", stroke_width=2, dash_length=0.15)

        ray2_dir = pt_S_star - pt_C2
        ray2_end = pt_C2 + 1.25 * ray2_dir
        ray2_arrow = Arrow(pt_C2, ray2_end, color=RAY, stroke_width=3, buff=0, tip_length=0.22)
        ray2_lbl = MathTex(r"\mathbf{u}_2", font_size=19, color=RAY).next_to(ray2_end, UR, buff=0.08)

        north_ref2 = DashedLine(pt_C2, pt_C2 + UP * 2.2, color=GREY, stroke_width=1.5, dash_length=0.12)
        ang2 = math.atan2(ray2_dir[1], ray2_dir[0])
        arc2 = Arc(radius=0.9, arc_center=pt_C2, start_angle=PI / 2, angle=-(PI / 2 - ang2), color=RAY, stroke_width=2)
        th_tex2 = MathTex(r"\theta_2", font_size=18, color=RAY).next_to(pt_C2 + 1.15 * np.array([math.cos((PI/2 + ang2)/2), math.sin((PI/2 + ang2)/2), 0]), UR, buff=0.05)

        ray3_dir = pt_S_star - pt_C3
        ray3_end = pt_C3 + 1.25 * ray3_dir
        ray3_arrow = Arrow(pt_C3, ray3_end, color=RAY, stroke_width=3, buff=0, tip_length=0.22)
        ray3_lbl = MathTex(r"\mathbf{u}_3", font_size=19, color=RAY).next_to(ray3_end, UP, buff=0.08)

        north_ref3 = DashedLine(pt_C3, pt_C3 + UP * 2.2, color=GREY, stroke_width=1.5, dash_length=0.12)
        ang3 = math.atan2(ray3_dir[1], ray3_dir[0])
        arc3 = Arc(radius=0.9, arc_center=pt_C3, start_angle=PI / 2, angle=-(PI / 2 - ang3), color=RAY, stroke_width=2)
        th_tex3 = MathTex(r"\theta_3", font_size=18, color=RAY).next_to(pt_C3 + 1.15 * np.array([math.cos((PI/2 + ang3)/2), math.sin((PI/2 + ang3)/2), 0]), UR, buff=0.05)

        dot_S_star = Dot(pt_S_star, color=SIGN, radius=0.16)
        ring_S_star = Circle(radius=0.32, color=SIGN, stroke_width=2.5).move_to(pt_S_star)
        lbl_S_star = MathTex(r"\mathbf{s}^* \text{ (True Sign Fix)}", font_size=21, color=SIGN).next_to(dot_S_star, UL, buff=0.2)

        self.play(
            Transform(title, title_multi),
            Transform(problem_formula, formula_intersect),
            FadeOut(slide_dot),
            FadeOut(slide_lbl),
            Create(road_path, run_time=0.5),
        )

        self.play(FadeIn(dot_C2), Write(lbl_C2), Create(north_ref2), run_time=0.6)
        self.play(GrowArrow(ray2_arrow), Write(ray2_lbl), run_time=0.6)
        self.play(Create(arc2), Write(th_tex2), run_time=0.5)

        self.play(FadeIn(dot_C3), Write(lbl_C3), Create(north_ref3), run_time=0.6)
        self.play(GrowArrow(ray3_arrow), Write(ray3_lbl), run_time=0.6)
        self.play(Create(arc3), Write(th_tex3), run_time=0.5)

        self.play(FadeIn(dot_S_star, scale=0.5), Create(ring_S_star), Write(lbl_S_star), run_time=0.8)
        self.play(ring_S_star.animate.scale(1.3).set_stroke(opacity=0.3), run_time=0.6)
        self.wait(1.5)

        title_lsq = MathTex(
            r"\text{Least Squares Estimation: } \mathbf{A}\mathbf{s} = \mathbf{b} \implies \mathbf{s}^* = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{b}",
            font_size=25, color=WHITE_C
        ).to_edge(UP, buff=0.45)

        geom_left = VGroup(
            dot_C1, lbl_C1, north_ref1, north_lbl1, total_arc1, th_tex1, ray_arrow, ray1_lbl_full,
            dot_C2, lbl_C2, north_ref2, arc2, th_tex2, ray2_arrow, ray2_lbl,
            dot_C3, lbl_C3, north_ref3, arc3, th_tex3, ray3_arrow, ray3_lbl,
            dot_S_star, ring_S_star, lbl_S_star, road_path, opt_axis, opt_axis_tag
        )

        mat_eq = MathTex(
            r"\underbrace{\begin{bmatrix} -\sin\theta_1 & \cos\theta_1 \\ -\sin\theta_2 & \cos\theta_2 \\ -\sin\theta_3 & \cos\theta_3 \end{bmatrix}}_{\mathbf{A}\;(N\times 2)}"
            r"\underbrace{\begin{bmatrix} x_s \\ y_s \end{bmatrix}}_{\mathbf{s}} = "
            r"\underbrace{\begin{bmatrix} -\sin\theta_1 x_1 + \cos\theta_1 y_1 \\ -\sin\theta_2 x_2 + \cos\theta_2 y_2 \\ -\sin\theta_3 x_3 + \cos\theta_3 y_3 \end{bmatrix}}_{\mathbf{b}}",
            font_size=19, color=WHITE_C
        ).move_to(RIGHT * 3.3 + UP * 0.7)

        note_xy = MathTex(
            r"(x_i,\, y_i) = \text{local-plane coords of camera } \mathbf{c}_i",
            font_size=17, color=GREY
        ).next_to(mat_eq, DOWN, buff=0.22)

        sol_box = MathTex(
            r"\mathbf{s}^* = (\mathbf{A}^T \mathbf{A})^{-1} \mathbf{A}^T \mathbf{b} = \begin{bmatrix} x_s^* \\ y_s^* \end{bmatrix}",
            font_size=22, color=FORM
        ).move_to(RIGHT * 3.3 + DOWN * 1.6)

        formula_lsq_bot = MathTex(
            r"\text{Overdetermined System: } N \ge 2 \text{ observations minimize bearing residual } \|\mathbf{A}\mathbf{s} - \mathbf{b}\|^2",
            font_size=22, color=FORM
        ).to_edge(DOWN, buff=0.45)

        self.play(
            Transform(title, title_lsq),
            Transform(problem_formula, formula_lsq_bot),
            geom_left.animate.scale(0.65).shift(LEFT * 3.2 + UP * 0.2),
            run_time=0.8
        )
        self.play(FadeIn(mat_eq, shift=LEFT * 0.3))
        self.play(FadeIn(note_xy, shift=UP * 0.1))
        self.play(FadeIn(sol_box, shift=UP * 0.2))
        self.wait(1.8)

        title_geo = MathTex(
            r"\text{Coordinate Conversion: Local Plane } \mathbf{s}^* \longrightarrow \text{WGS84 GPS } (\phi_s,\, \lambda_s)",
            font_size=25, color=WHITE_C
        ).to_edge(UP, buff=0.45)

        geo_eq = MathTex(
            r"\begin{aligned}"
            r"\phi_s &= \phi_0 + \frac{y_s^*}{R_M} \cdot \frac{180}{\pi} \\[6pt]"
            r"\lambda_s &= \lambda_0 + \frac{x_s^*}{R_N \cos\phi_0} \cdot \frac{180}{\pi}"
            r"\end{aligned}",
            font_size=24, color=WHITE_C
        ).move_to(RIGHT * 2.2 + UP * 0.2)

        gps_badge = VGroup(
            RoundedRectangle(corner_radius=0.12, width=4.5, height=1.1, fill_color="#13233a", fill_opacity=0.9, stroke_color=FORM, stroke_width=2),
            MathTex(r"\mathbf{s}^* \text{ GPS: } (10.7769^\circ\text{N},\, 106.7009^\circ\text{E})", font_size=22, color=FORM)
        ).move_to(RIGHT * 2.2 + DOWN * 1.5)

        formula_geo_bot = MathTex(
            r"R_M, R_N: \text{Meridian and Prime Vertical radii of curvature of WGS84 Ellipsoid}",
            font_size=22, color=FORM
        ).to_edge(DOWN, buff=0.45)

        self.play(
            Transform(title, title_geo),
            Transform(problem_formula, formula_geo_bot),
            FadeOut(mat_eq), FadeOut(sol_box), FadeOut(note_xy),
            FadeIn(geo_eq, shift=UP * 0.2),
            FadeIn(gps_badge, shift=UP * 0.2),
            run_time=0.8
        )
        self.wait(2.5)



if __name__ == "__main__":
    import subprocess
    import sys
    args = sys.argv[1:] if len(sys.argv) > 1 else ["-pqm"]
    subprocess.run([sys.executable, "-m", "manim", *args, __file__, "SignGPSExplainer"])

