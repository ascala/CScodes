import sympy as sp # Symbolic computation library
import numpy as np # Numerical computation library

def Define_Potential():

    # Define the potential V(x) and the derivative dV/dx
    x = sp.symbols("x") # define variables and parameters as symnbolic objects
    exprV = (x/np.pi)**2 - sp.cos(2*x) # symbolic expression for the potential V(x)
    dexprV = sp.diff(exprV, x) # symbolic expression for the derivative dV/dx

    # Get the variables in the expression
    variables = sorted(exprV.free_symbols, key=lambda x: x.name)

    # Create lambda functions from the SymPy expression using NumPy
    V = sp.lambdify(variables, exprV, 'numpy')  # numerical function for V(x)
    dV_dx = sp.lambdify(variables, dexprV, 'numpy')  # numerical function for dV/dx

    return V,dV_dx
