import turtle
import random
import tkinter

# Explicitly set the Tcl/Tk version
tkinter.TkVersion = 8.6
tkinter.TclVersion = 8.6

# Set up the turtle
screen = turtle.Screen()
screen.bgcolor("white")

artist = turtle.Turtle()
artist.speed(0)
artist.hideturtle()

# Function to draw a polygon with random color
def draw_polygon(sides, length):
    artist.fillcolor(random.choice(["red", "blue", "green", "yellow", "purple", "orange"]))
    artist.begin_fill()
    for _ in range(sides):
        artist.forward(length)
        artist.right(360 / sides)
    artist.end_fill()

# Function to draw low poly art
def draw_low_poly_art(num_polygons, min_sides, max_sides, min_length, max_length):
    for _ in range(num_polygons):
        sides = random.randint(min_sides, max_sides)
        length = random.uniform(min_length, max_length)
        x = random.randint(-300, 300)
        y = random.randint(-300, 300)

        artist.penup()
        artist.goto(x, y)
        artist.pendown()

        draw_polygon(sides, length)

# Set parameters for low poly art
num_polygons = 10
min_sides = 3
max_sides = 6
min_length = 20
max_length = 100

# Draw low poly art
draw_low_poly_art(num_polygons, min_sides, max_sides, min_length, max_length)

# Close the window on click
screen.exitonclick()