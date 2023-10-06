import ipywidgets as wg
from IPython.display import SVG

def f(x):
    if x == 1:
        return SVG(filename='./path/to/pic1')
    elif x == 2:
        return SVG(filename='./path/to/pic2')
    elif x == 3:
        return SVG(filename='./path/to/pic3')
    else:
        return SVG(filename='./path/to/pic4')

if __name__ == '__main__':
    wg.interact(f, x=wg.IntSlider(min=1,max=4,step=1))
