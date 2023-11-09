"""
galerkin.py - A Python module for solving partial differential equations using Galerkin methods.

This module provides a collection of classes and functions for solving partial differential equations (PDEs)
using Galerkin methods. It includes various function spaces, basis functions, and tools for assembling matrices
and solving linear systems arising from finite element discretizations.

Contents:
- FunctionSpace: Base class for defining function spaces.
- Legendre: Function space based on Legendre polynomials.
- Chebyshev: Function space based on Chebyshev polynomials.
- Sines: Function space for trigonometric functions (sines).
- Cosines: Function space for trigonometric functions (cosines).
- Dirichlet: Class for handling Dirichlet boundary conditions.
- Neumann: Class for handling Neumann boundary conditions.
- Composite: Base class for function spaces created as linear combinations of orthogonal basis functions.
- DirichletLegendre: Function space with Dirichlet boundary conditions based on Legendre polynomials.
- DirichletChebyshev: Function space with Dirichlet boundary conditions based on Chebyshev polynomials.
- NeumannLegendre: Function space with Neumann boundary conditions based on Legendre polynomials (not implemented).
- NeumannChebyshev: Function space with Neumann boundary conditions based on Chebyshev polynomials (not implemented).
- BasisFunction: Base class for defining basis functions.
- TestFunction: Basis function for testing purposes.
- TrialFunction: Basis function for trial purposes.
- assemble_generic_matrix: Assemble a generic matrix using basis functions.
- inner: Compute the inner product between basis functions.
- project: Project a given function onto a function space.
- L2_error: Compute the L2 error between two functions.
- test_project: Test function for projecting a function onto various function spaces.
- test_helmholtz: Test function for solving the Helmholtz equation with different function spaces.
- test_convection_diffusion: Test function for solving the convection-diffusion problem with different function spaces.
"""
import numpy as np
import sympy as sp
import scipy.sparse as sparse
from scipy.integrate import quad
from numpy.polynomial import Legendre as Leg
from numpy.polynomial import Chebyshev as Cheb

x = sp.Symbol('x')


def map_reference_domain(x, d, r):
    """
    Map a point 'x' from the true domain to the reference domain.

    Parameters:
    - x (float or numpy.ndarray): The point(s) in the true domain to be mapped.
    - d (tuple): The domain of the true space as a tuple (d_min, d_max).
    - r (tuple): The reference domain as a tuple (r_min, r_max).

    Returns:
    - float or numpy.ndarray: The point(s) mapped to the reference domain.

    This function maps a point or array of points 'x' from the true domain (defined by 'd')
    to the reference domain (defined by 'r'). It is useful in coordinate transformations
    for numerical simulations and finite element analysis.
    """
    return r[0] + (r[1]-r[0])*(x-d[0])/(d[1]-d[0])


def map_true_domain(x, d, r):
    """
    Map a point 'x' from the reference domain to the true domain.

    Parameters:
    - x (float or numpy.ndarray): The point(s) in the reference domain to be mapped.
    - d (tuple): The domain of the true space as a tuple (d_min, d_max).
    - r (tuple): The reference domain as a tuple (r_min, r_max).

    Returns:
    - float or numpy.ndarray: The point(s) mapped to the true domain.

    This function maps a point or array of points 'x' from the reference domain (defined by 'r')
    to the true domain (defined by 'd'). It is useful in coordinate transformations
    for numerical simulations and finite element analysis.
    """
    return d[0] + (d[1]-d[0])*(x-r[0])/(r[1]-r[0])


def map_expression_true_domain(u, x, d, r):
    """
    Map an expression or function 'u' from the true domain to the reference domain.

    Parameters:
    - u (str or sympy.Expr): The expression or function to be mapped.
    - x (sympy.Symbol): The symbolic variable representing the spatial coordinate.
    - d (tuple): The domain of the true space as a tuple (d_min, d_max).
    - r (tuple): The reference domain as a tuple (r_min, r_max).

    Returns:
    - sympy.Expr: The expression 'u' with coordinates mapped to the reference domain.

    This function maps an expression or function 'u' with spatial coordinate 'x' from the true domain (defined by 'd')
    to the reference domain (defined by 'r'). It is useful for transforming mathematical expressions or functions
    to work in a different coordinate system, typically used in finite element analysis and numerical simulations.
    """
    if d != r:
        u = sp.sympify(u)
        xm = map_true_domain(x, d, r)
        u = u.replace(x, xm)
    return u


class FunctionSpace:
    """
    A class representing a function space with basis functions.

    This class provides a framework for working with function spaces defined by a set of basis functions.
    Function spaces are commonly used in numerical simulations, finite element analysis, and mathematical computations.

    Attributes:
    - N (int): The number of basis functions in the function space.
    - _domain (tuple): A tuple representing the domain of the function space.

    Properties:
    - domain (tuple): Get the domain of the function space.
    - reference_domain: Get the reference domain (if implemented) of the function space.
    - domain_factor: Calculate the domain factor, which scales the domain relative to the reference domain.

    Methods:
    - mesh(N=None): Generate a mesh of points within the domain.
    - weight(x=x): Calculate the weight at a given point (default is 1).
    - basis_function(j, sympy=False): Get the j-th basis function (possibly symbolic if sympy=True).
    - derivative_basis_function(j, k=1): Get the k-th derivative of the j-th basis function.
    - evaluate_basis_function(Xj, j): Evaluate the j-th basis function at a given point(s).
    - evaluate_derivative_basis_function(Xj, j, k=1): Evaluate the k-th derivative of the j-th basis function.
    - eval(uh, xj): Evaluate the function represented by 'uh' at the specified point(s).
    - eval_basis_function_all(Xj): Evaluate all basis functions at the specified point(s).
    - eval_derivative_basis_function_all(Xj, k=1): Evaluate all derivatives of basis functions at the specified point(s).
    - inner_product(u): Calculate the inner product of a function 'u' with respect to the basis functions.
    - mass_matrix(): Calculate the mass matrix of the function space.

    Notes:
    - This class is a foundational component for working with function spaces.
    - It provides methods for generating meshes, evaluating basis functions, calculating inner products, and more.
    - Some methods are intended to be implemented in derived classes or based on specific requirements.

    Examples:
    - To create a FunctionSpace object: fs = FunctionSpace(N=10, domain=(-1, 1))
    - To evaluate a basis function: value = fs.evaluate_basis_function(x, j)
    - To calculate the mass matrix: M = fs.mass_matrix()

    """
    def __init__(self, N, domain=(-1, 1)):
        """
        Initialize a FunctionSpace object.

        Parameters:
        - N (int): The number of basis functions.
        - domain (tuple, optional): A tuple representing the domain of the function space.
          Defaults to (-1, 1).

        This constructor sets the number of basis functions and the domain of the function space.
        """
        self.N = N
        self._domain = domain

    @property
    def domain(self):
        """
        Get the domain of the FunctionSpace.

        Returns:
        - tuple: A tuple representing the domain of the function space.

        This property allows you to retrieve the domain (a tuple) of the FunctionSpace.
        """
        return self._domain

    @property
    def reference_domain(self):
        """
        Get the reference domain of the FunctionSpace.

        Returns:
        - tuple: The reference domain for the function space, typically (-1, 1).

        This property returns the reference domain for the function space, which is used for mapping the physical domain
        of elements to a standardized reference domain. The reference domain is commonly (-1, 1).
        """
        return (-1, 1)

    @property
    def domain_factor(self):
        """
        Calculate the domain factor.

        Returns:
        - float: The domain factor.

        The domain factor represents the scaling factor for the current domain
        with respect to the reference domain.

        It is calculated as the ratio of the width of the current domain to
        the width of the reference domain.
        """
        d = self.domain
        r = self.reference_domain
        return (d[1]-d[0])/(r[1]-r[0])

    def mesh(self, N=None):
        """
        Generate a mesh of points within the domain.

        Parameters:
        - N (int, optional): The number of mesh points. If not provided, it defaults to the 'N' value set during initialization.

        Returns:
        - numpy.ndarray: An array of mesh points.

        This method generates a mesh of points within the domain of the FunctionSpace.
        It divides the domain into 'N+1' equally spaced points if 'N' is provided or 'self.N+1' points if 'N' is not provided.
        The mesh points are returned as a NumPy array.
        """
        d = self.domain
        n = N if N is not None else self.N
        return np.linspace(d[0], d[1], n+1)

    def weight(self, x=x):
        """
        Calculate the weight at a given point.

        Parameters:
        - x (float, optional): The point at which to calculate the weight. Defaults to 'x'.

        Returns:
        - int: The weight value.

        This method calculates and returns the weight at the specified point 'x'.
        The default weight value is 1, but you can override it by providing a different 'x' value.
        """
        return 1

    def basis_function(self, j, sympy=False):
        """
        Get the j-th basis function.

        Parameters:
        - j (int): The index of the basis function to retrieve.
        - sympy (bool, optional): If True, return a symbolic expression using SymPy.
                                If False (default), return a placeholder indicating that the basis function is not implemented.

        Returns:
        - function or sympy.Expr: The j-th basis function.

        This method is intended to retrieve the j-th basis function of the FunctionSpace.
        By default, it returns a placeholder indicating that the basis function is not implemented.
        If 'sympy' is set to True, it returns a symbolic expression (using SymPy) representing the basis function.
        """
        if sympy:
            return sp.legendre(j, x)
        else:
            return Legendre.basis_function(j)

    def derivative_basis_function(self, j, k=1):
        """
        Get the k-th derivative of the j-th basis function.

        Parameters:
        - j (int): The index of the basis function for which to calculate the derivative.
        - k (int, optional): The order of the derivative to calculate. Defaults to 1.

        Returns:
        - function: A function representing the k-th derivative of the j-th basis function.

        This method is intended to calculate and return a function representing the k-th derivative
        of the j-th basis function of the FunctionSpace. You may need to implement the derivative
        functions separately based on your specific application.
        """
        return BasisFunction(j).deriv(k)

    def evaluate_basis_function(self, Xj, j):
        """
        Evaluate the j-th basis function at a given point.

        Parameters:
        - Xj (float or numpy.ndarray): The point or points at which to evaluate the basis function.
        - j (int): The index of the basis function to evaluate.

        Returns:
        - float or numpy.ndarray: The value of the j-th basis function at the specified point(s).

        This method is used to evaluate the j-th basis function of the FunctionSpace
        at one or more points specified by 'Xj'. The result is a single value or an array
        of values representing the evaluation of the basis function at the given point(s).
        """
        return self.basis_function(j)(Xj)

    def evaluate_derivative_basis_function(self, Xj, j, k=1):
        """
        Evaluate the k-th derivative of the j-th basis function at a given point.

        Parameters:
        - Xj (float or numpy.ndarray): The point or points at which to evaluate the derivative of the basis function.
        - j (int): The index of the basis function for which to calculate the derivative.
        - k (int, optional): The order of the derivative to calculate. Defaults to 1.

        Returns:
        - float or numpy.ndarray: The value of the k-th derivative of the j-th basis function at the specified point(s).

        This method is used to evaluate the k-th derivative of the j-th basis function of the FunctionSpace
        at one or more points specified by 'Xj'. The result is a single value or an array
        of values representing the evaluation of the derivative at the given point(s).
        """
        return self.derivative_basis_function(j, k=k)(Xj)

    def eval(self, uh, xj):
        """
        Evaluate the function represented by 'uh' at the specified point(s).

        Parameters:
        - uh (numpy.ndarray): An array representing the coefficients of the function to be evaluated.
        - xj (float or numpy.ndarray): The point or points at which to evaluate the function.

        Returns:
        - float or numpy.ndarray: The value(s) of the function represented by 'uh' at the specified point(s).

        This method is used to evaluate the function represented by the coefficients 'uh'
        at the specified point(s) 'xj'. It performs the evaluation using the basis functions
        defined within the FunctionSpace.
        """
        xj = np.atleast_1d(xj)
        Xj = map_reference_domain(xj, self.domain, self.reference_domain)
        P = self.eval_basis_function_all(Xj)
        return P @ uh

    def eval_basis_function_all(self, Xj):
        """
        Evaluate all basis functions at the specified point(s).

        Parameters:
        - Xj (float or numpy.ndarray): The point or points at which to evaluate all basis functions.

        Returns:
        - numpy.ndarray: An array of shape (N+1, ) representing the values of all basis functions
                        at the specified point(s).

        This method is used to evaluate all the basis functions defined within the FunctionSpace
        at the specified point(s) 'Xj'. It returns an array containing the values of all basis functions
        at the given point(s).
        """
        P = np.zeros((len(Xj), self.N+1))
        for j in range(self.N+1):
            P[:, j] = self.evaluate_basis_function(Xj, j)
        return P

    def eval_derivative_basis_function_all(self, Xj, k=1):
        """
        Evaluate all derivatives of basis functions at the specified point(s).

        Parameters:
        - Xj (float or numpy.ndarray): The point or points at which to evaluate the derivatives of basis functions.
        - k (int, optional): The order of the derivatives to calculate. Defaults to 1.

        Returns:
        - numpy.ndarray: An array of shape (len(Xj), N+1) representing the values of all derivatives
                        of basis functions at the specified point(s).

        This method is intended to evaluate all derivatives of basis functions up to the k-th order
        at the specified point(s) 'Xj'. The result is an array containing the values of all derivatives
        of basis functions at the given point(s).
        """
        raise NotImplementedError

    def inner_product(self, u):
        """
        Calculate the inner product of a function 'u' with respect to the basis functions.

        Parameters:
        - u: A function or expression to calculate the inner product with.

        Returns:
        - numpy.ndarray: An array of shape (N+1, ) representing the inner product values.

        This method calculates the inner product of a given function 'u' with respect to the
        basis functions defined within the FunctionSpace. It returns an array containing the
        inner product values for each basis function.
        """
        us = map_expression_true_domain(
            u, x, self.domain, self.reference_domain)
        us = sp.lambdify(x, us)
        uj = np.zeros(self.N+1)
        h = self.domain_factor
        r = self.reference_domain
        for i in range(self.N+1):
            psi = self.basis_function(i)
            def uv(Xj): return us(Xj) * psi(Xj)
            uj[i] = float(h) * quad(uv, float(r[0]), float(r[1]))[0]
        return uj

    def mass_matrix(self):
        """
        Calculate the mass matrix of the FunctionSpace.

        Returns:
        - numpy.ndarray: A 2D array representing the mass matrix.

        This method calculates and returns the mass matrix of the FunctionSpace.
        The mass matrix is often used in numerical simulations and finite element analysis
        to represent the influence of basis functions on the system's behavior.
        """
        return assemble_generic_matrix(TrialFunction(self), TestFunction(self))


class Legendre(FunctionSpace):
    """
    A class representing a Legendre polynomial basis function space.

    This class is a specialized extension of the FunctionSpace class that focuses on Legendre polynomial basis functions.
    Legendre polynomials are commonly used as basis functions in numerical simulations, especially in finite element analysis.

    Attributes:
    - N (int): The number of Legendre polynomial basis functions.
    - domain (tuple): The domain of the Legendre polynomial space, typically (-1, 1).

    Methods:
    - basis_function(j, sympy=False): Get the j-th Legendre polynomial basis function.
    - derivative_basis_function(j, k=1): Get the k-th derivative of the j-th Legendre polynomial basis function.
    - L2_norm_sq(N): Calculate the square of the L2 norm of the Legendre polynomial basis functions.
    - mass_matrix(): Calculate the mass matrix of the Legendre polynomial basis functions.
    - eval(uh, xj): Evaluate the function represented by 'uh' at the specified point(s) 'xj' using Legendre basis functions.

    Notes:
    - Legendre polynomials are orthogonal polynomials commonly used for approximating functions in various applications.
    - This class provides methods for working with Legendre polynomial basis functions within a specified domain.

    Examples:
    - To create a Legendre object: leg_space = Legendre(N=10, domain=(-1, 1))
    - To evaluate a Legendre basis function: basis_val = leg_space.basis_function(j, sympy=True)
    - To calculate the mass matrix: M = leg_space.mass_matrix()

    """
    def __init__(self, N, domain=(-1, 1)):
        """
        Initialize a Legendre polynomial basis function space.

        Parameters:
        - N (int): The number of Legendre polynomial basis functions.
        - domain (tuple, optional): The domain of the Legendre polynomial space. Defaults to (-1, 1).

        This constructor sets the number of basis functions and the domain of the Legendre polynomial space.
        """
        FunctionSpace.__init__(self, N, domain=domain)

    def basis_function(self, j, sympy=False):
        """
        Get the j-th Legendre basis function.

        Parameters:
        - j (int): The index (degree) of the basis function to retrieve.
        - sympy (bool, optional): If True, returns a symbolic expression using sympy.
                                  If False (default), returns a numerical representation using numpy.

        Returns:
        - Leg or sp.Expr: The j-th Legendre basis function as a numpy polynomial Legendre instance
                          or a sympy expression, depending on the 'sympy' flag.

        The method retrieves the j-th Legendre basis function of the function space. If the 'sympy'
        parameter is True, it returns a symbolic expression using sympy, which can be used for
        symbolic computations. Otherwise, it returns a numerical representation as a numpy polynomial
        Legendre instance, which can be used for numerical evaluations and operations.

        Legendre polynomials are orthogonal polynomials with respect to the weight function w(x) = 1
        on the interval [-1, 1]. They are widely used in numerical analysis for polynomial approximation
        and in solving differential equations.
        """
        if sympy:
            # Use sympy to return a symbolic expression for the j-th Legendre polynomial
            return sp.legendre(j, x)
        else:
            # Use numpy to return the j-th Legendre polynomial as a numpy polynomial object
            return Leg.basis(j)

    def derivative_basis_function(self, j, k=1):
        """
        Get the k-th derivative of the j-th Legendre polynomial basis function.

        Parameters:
        - j (int): The index of the Legendre polynomial basis function for which to calculate the derivative.
        - k (int, optional): The order of the derivative to calculate. Defaults to 1.

        Returns:
        - function or sympy.Expr: The k-th derivative of the j-th Legendre polynomial basis function.

        This method is used to calculate and return the k-th derivative of the j-th Legendre polynomial basis function
        within the Legendre polynomial space. It provides flexibility by allowing you to choose between returning
        a callable function (default) or a symbolic expression (using SymPy) representing the derivative.
        """
        return self.basis_function(j).deriv(k)

    def L2_norm_sq(self, N):
        """
        Calculate the square of the L2 norm of the Legendre polynomial basis functions.

        Parameters:
        - N (int): The number of Legendre polynomial basis functions to include in the calculation.

        Returns:
        - float: The square of the L2 norm of the Legendre polynomial basis functions.

        This method calculates the square of the L2 norm of the Legendre polynomial basis functions up to the N-th basis function.
        The L2 norm quantifies the magnitude of these basis functions within the function space. This operation is useful
        for assessing the orthogonality and scaling of Legendre polynomials in numerical computations.
        """
        norm_sq_sum = 0
        for j in range(N):  # Assuming self.N is the number of basis functions
            # Get the j-th Legendre basis function
            Pj = self.basis_function(j, sympy=False)
            # Integrate Pj(x)^2 from -1 to 1
            norm_sq, _ = quad(lambda x: Pj(x)**2, -1, 1)
            norm_sq_sum += norm_sq
        return norm_sq_sum


    def mass_matrix(self):
        """
        Calculate the mass matrix of the Legendre polynomial basis functions.

        Returns:
        - numpy.ndarray: A 2D array representing the mass matrix.

        This method calculates and returns the mass matrix of the Legendre polynomial basis functions within the function space.
        The mass matrix represents the influence of these basis functions on the system's behavior and is commonly used
        in numerical simulations and finite element analysis for solving partial differential equations and other tasks.
        """
        diagonal_elements = [self.basis_function(i) for i in range(self.N)]
        return np.diag(diagonal_elements)



    def eval(self, uh, xj):
        """
        Evaluate the function represented by 'uh' at the specified point(s) 'xj' using Legendre basis functions.

        Parameters:
        - uh (numpy.ndarray): An array representing the coefficients of the function to be evaluated.
        - xj (float or numpy.ndarray): The point or points at which to evaluate the function.

        Returns:
        - float or numpy.ndarray: The value(s) of the function represented by 'uh' at the specified point(s).

        This method is used to evaluate the function represented by the coefficients 'uh' at the specified point(s) 'xj'
        using Legendre basis functions. It performs the evaluation and returns the value(s) of the function at the specified
        point(s). The output can be a single value or an array of values, depending on the input 'xj'.

        Example:
        - To evaluate a function represented by 'uh' at a point 'x': value = leg_space.eval(uh, x)
        """
        xj = np.atleast_1d(xj)
        Xj = map_reference_domain(xj, self.domain, self.reference_domain)
        return np.polynomial.legendre.legval(Xj, uh)



class Chebyshev(FunctionSpace):
    """
    A class representing a Chebyshev polynomial basis function space.

    This class is a specialized extension of the FunctionSpace class that focuses on Chebyshev polynomial basis functions.
    Chebyshev polynomials are commonly used as basis functions in numerical simulations and approximation tasks.

    Attributes:
    - N (int): The number of Chebyshev polynomial basis functions.
    - domain (tuple): The domain of the Chebyshev polynomial space, typically (-1, 1).

    Methods:
    - basis_function(j, sympy=False): Get the j-th Chebyshev polynomial basis function.
    - derivative_basis_function(j, k=1): Get the k-th derivative of the j-th Chebyshev polynomial basis function.
    - weight(x=x): Calculate the weight function of the Chebyshev polynomial space.
    - L2_norm_sq(N): Calculate the square of the L2 norm of the Chebyshev polynomial basis functions.
    - mass_matrix(): Calculate the mass matrix of the Chebyshev polynomial basis functions.
    - eval(uh, xj): Evaluate the function represented by 'uh' at the specified point(s) 'xj' using Chebyshev basis functions.
    - inner_product(u): Calculate the inner product of a function 'u' with respect to the Chebyshev polynomial basis functions.

    Notes:
    - Chebyshev polynomials are orthogonal polynomials widely used for various numerical and mathematical computations.
    - This class provides methods for working with Chebyshev polynomial basis functions within a specified domain.

    Examples:
    - To create a Chebyshev object: cheb_space = Chebyshev(N=10, domain=(-1, 1))
    - To evaluate a Chebyshev basis function: basis_val = cheb_space.basis_function(j, sympy=True)
    - To calculate the mass matrix: M = cheb_space.mass_matrix()

    """
    def __init__(self, N, domain=(-1, 1)):
        """
        Initialize a Chebyshev polynomial basis function space.

        Parameters:
        - N (int): The number of Chebyshev polynomial basis functions.
        - domain (tuple, optional): The domain of the Chebyshev polynomial space. Defaults to (-1, 1).

        This constructor sets the number of basis functions and the domain of the Chebyshev polynomial space.
        """
        FunctionSpace.__init__(self, N, domain=domain)

    def basis_function(self, j, sympy=False):
        """
        Get the j-th Chebyshev polynomial basis function.

        Parameters:
        - j (int): The index of the Chebyshev polynomial basis function to retrieve.
        - sympy (bool, optional): If True, return a symbolic expression using SymPy.
                                If False (default), return a callable function representing the basis function.

        Returns:
        - function or sympy.Expr: The j-th Chebyshev polynomial basis function.

        This method is used to retrieve the j-th Chebyshev polynomial basis function within the Chebyshev polynomial space.
        By default, it returns a callable function representing the basis function. If 'sympy' is set to True, it returns
        a symbolic expression (using SymPy) representing the basis function.

        Example:
        - To evaluate a Chebyshev basis function: basis_val = cheb_space.basis_function(j, sympy=True)
        """
        if sympy:
            # Return a symbolic expression for the Chebyshev polynomial basis function (if implemented using SymPy).
            return sp.cos(j * sp.acos(x))
        else:
            # Return a callable function representing the Chebyshev polynomial basis function.
            return Cheb.basis(j)

    def derivative_basis_function(self, j, k=1):
        """
        Get the k-th derivative of the j-th Chebyshev polynomial basis function.

        Parameters:
        - j (int): The index of the Chebyshev polynomial basis function for which to calculate the derivative.
        - k (int, optional): The order of the derivative to calculate. Defaults to 1.

        Returns:
        - function or sympy.Expr: The k-th derivative of the j-th Chebyshev polynomial basis function.

        This method is used to calculate and return the k-th derivative of the j-th Chebyshev polynomial basis function
        within the Chebyshev polynomial space. It provides flexibility by allowing you to choose between returning
        a callable function (default) or a symbolic expression (using SymPy) representing the derivative.
        """
        return self.basis_function(j).deriv(k)

    def weight(self, x=x):
        """
        Calculate the weight function of the Chebyshev polynomial basis functions.

        Parameters:
        - x (sympy.Symbol, optional): The symbolic variable representing the spatial coordinate. Defaults to 'x'.

        Returns:
        - sympy.Expr: The weight function of the Chebyshev polynomial basis functions.

        This method calculates and returns the weight function associated with the Chebyshev polynomial basis functions.
        The weight function is typically used in orthogonal polynomial systems and is essential for performing various
        numerical computations and transformations within the Chebyshev polynomial space.
        """
        return 1/sp.sqrt(1-x**2)

    def L2_norm_sq(self, N):
        """
        Calculate the square of the L2 norm of the Chebyshev polynomial basis functions.

        Parameters:
        - N (int): The number of Chebyshev polynomial basis functions to include in the calculation.

        Returns:
        - float: The square of the L2 norm of the Chebyshev polynomial basis functions.

        This method calculates the square of the L2 norm of the Chebyshev polynomial basis functions up to the N-th basis function.
        The L2 norm quantifies the magnitude of these basis functions within the function space. This operation is useful
        for assessing the orthogonality and scaling of Chebyshev polynomials in numerical computations.

        Note:
        - The implementation of this method is not provided (raises NotImplementedError).

        """
        x = sp.symbols('x')
        # Start with the square of the 0th polynomial
        norm_sq_0 = sp.integrate((self.basis_function(0, sympy=True)**2) * self.weight(x), (x, -1, 1))
        norm_sq_sum = norm_sq_0
        # Sum the squared L2 norms of the rest of the Chebyshev polynomials
        for j in range(1, N):
            norm_sq_j = sp.integrate((self.basis_function(j, sympy=True)**2) * self.weight(x), (x, -1, 1))
            norm_sq_sum += norm_sq_j
        return float(norm_sq_sum)

    def mass_matrix(self):
        """
        Calculate the mass matrix of the Chebyshev polynomial basis functions.

        Returns:
        - numpy.ndarray: A 2D array representing the mass matrix.

        This method calculates and returns the mass matrix of the Chebyshev polynomial basis functions within the function space.
        The mass matrix represents the influence of these basis functions on the system's behavior and is commonly used
        in numerical simulations and finite element analysis for solving partial differential equations and other tasks.

        Note:
        - The implementation of this method is not provided (raises NotImplementedError).

        """
        raise NotImplementedError("mass_matrix is not implemented.")


    def eval(self, uh, xj):
        """
        Evaluate the function represented by 'uh' at the specified point(s) 'xj' using Chebyshev basis functions.

        Parameters:
        - uh (numpy.ndarray): An array representing the coefficients of the function to be evaluated.
        - xj (float or numpy.ndarray): The point or points at which to evaluate the function.

        Returns:
        - float or numpy.ndarray: The value(s) of the function represented by 'uh' at the specified point(s).

        This method is used to evaluate the function represented by the coefficients 'uh' at the specified point(s) 'xj'
        using Chebyshev basis functions. It performs the evaluation and returns the value(s) of the function at the specified
        point(s). The output can be a single value or an array of values, depending on the input 'xj'.

        Example:
        - To evaluate a function represented by 'uh' at a point 'x': value = cheb_space.eval(uh, x)
        """
        xj = np.atleast_1d(xj)
        Xj = map_reference_domain(xj, self.domain, self.reference_domain)
        return np.polynomial.chebyshev.chebval(Xj, uh)

    def inner_product(self, u):
        """
        Calculate the inner product of a function 'u' with respect to the Chebyshev polynomial basis functions.

        Parameters:
        - u (sympy.Expr or str): The function 'u' to calculate the inner product for.

        Returns:
        - numpy.ndarray: An array representing the inner product of 'u' with the Chebyshev polynomial basis functions.

        This method calculates the inner product of the input function 'u' with respect to the Chebyshev polynomial basis functions
        within the Chebyshev polynomial space. The result is returned as an array, where each element corresponds to the
        inner product with a specific basis function.

        Notes:
        - The function 'u' should be expressed as a SymPy expression or a string that can be converted to a SymPy expression.
        - The inner product calculation is performed using numerical integration techniques.

        Example:
        - To calculate the inner product of 'u' with respect to the Chebyshev basis functions:
        inner_prod = cheb_space.inner_product(u)
        """
        us = map_expression_true_domain(
            u, x, self.domain, self.reference_domain)
        # Change of variables to x=cos(theta)
        us = sp.simplify(us.subs(x, sp.cos(x)), inverse=True)
        us = sp.lambdify(x, us)
        uj = np.zeros(self.N+1)
        h = float(self.domain_factor)
        k = sp.Symbol('k')
        basis = sp.lambdify((k, x), sp.simplify(
            self.basis_function(k, True).subs(x, sp.cos(x), inverse=True)))
        for i in range(self.N+1):
            def uv(Xj, j): return us(Xj) * basis(j, Xj)
            uj[i] = float(h) * quad(uv, 0, np.pi, args=(i,))[0]
        return uj

class Trigonometric(FunctionSpace):
    """
    Base class for trigonometric function spaces.

    This class serves as a base class for function spaces based on trigonometric basis functions. Trigonometric functions
    are often used for various numerical simulations and mathematical computations.

    Attributes (inherited from FunctionSpace):
    - N (int): The number of basis functions.
    - domain (tuple): The domain of the trigonometric function space.

    Methods:
    - reference_domain: Get the reference domain for trigonometric functions.
    - mass_matrix: Calculate the mass matrix of the trigonometric basis functions.
    - eval(uh, xj): Evaluate the function represented by 'uh' at the specified point(s) 'xj' using trigonometric basis functions.

    Notes:
    - Trigonometric function spaces are commonly used in tasks involving periodic functions and Fourier analysis.
    - This class provides methods for working with trigonometric basis functions within a specified domain.

    Examples:
    - To create a Trigonometric object: trig_space = Trigonometric(N=10, domain=(0, 2*np.pi))
    - To calculate the mass matrix: M = trig_space.mass_matrix()
    - To evaluate a function represented by 'uh' at a point 'x': value = trig_space.eval(uh, x)
    """
    
    @property
    def reference_domain(self):
        """
        Get the reference domain for trigonometric functions.

        Returns:
        - tuple: The reference domain for trigonometric functions, typically (0, 1).

        This property returns the reference domain for trigonometric functions, which is often used as the default
        domain when working with trigonometric basis functions. The reference domain is commonly (0, 1).
        """
        return (0, 1)

    def mass_matrix(self):
        """
        Calculate the mass matrix of the trigonometric basis functions.

        Returns:
        - scipy.sparse.csr_matrix: A sparse matrix representing the mass matrix.

        This method calculates and returns the mass matrix of the trigonometric basis functions within the function space.
        The mass matrix is an essential component for solving various numerical simulations and partial differential
        equations involving trigonometric functions.
        """
        return sparse.diags([self.L2_norm_sq(self.N+1)], [0], (self.N+1, self.N+1), format='csr')

    def eval(self, uh, xj):
        """
        Evaluate the function represented by 'uh' at the specified point(s) 'xj' using trigonometric basis functions.

        Parameters:
        - uh (numpy.ndarray): An array representing the coefficients of the function to be evaluated.
        - xj (float or numpy.ndarray): The point or points at which to evaluate the function.

        Returns:
        - float or numpy.ndarray: The value(s) of the function represented by 'uh' at the specified point(s).

        This method is used to evaluate the function represented by the coefficients 'uh' at the specified point(s) 'xj'
        using trigonometric basis functions. It performs the evaluation and returns the value(s) of the function at the
        specified point(s). The output can be a single value or an array of values, depending on the input 'xj'.

        Example:
        - To evaluate a function represented by 'uh' at a point 'x': value = trig_space.eval(uh, x)
        """
        xj = np.atleast_1d(xj)
        Xj = map_reference_domain(xj, self.domain, self.reference_domain)
        P = self.eval_basis_function_all(Xj)
        return P @ uh + self.B.Xl(Xj)


class Sines(Trigonometric):
    """
    Class representing a function space based on sine basis functions.

    This class extends the Trigonometric class and represents a function space based on sine basis functions.
    Sine functions are commonly used in various mathematical and numerical applications, including Fourier analysis.

    Attributes (inherited from Trigonometric):
    - N (int): The number of basis functions.
    - domain (tuple): The domain of the sine function space.

    Methods (inherited from Trigonometric):
    - reference_domain: Get the reference domain for sine functions.
    - mass_matrix: Calculate the mass matrix of the sine basis functions.
    - eval(uh, xj): Evaluate the function represented by 'uh' at the specified point(s) 'xj' using sine basis functions.

    Additional Methods:
    - basis_function(j, sympy=False): Get the j-th sine basis function.
    - derivative_basis_function(j, k=1): Get the k-th derivative of the j-th sine basis function.
    - L2_norm_sq(N): Calculate the square of the L2 norm of sine basis functions.

    Example:
    - To create a Sines object: sine_space = Sines(N=10, domain=(0, 2*np.pi))
    - To calculate the mass matrix: M = sine_space.mass_matrix()
    - To evaluate a function represented by 'uh' at a point 'x': value = sine_space.eval(uh, x)
    """
    
    def __init__(self, N, domain=(0, 1), bc=(0, 0)):
        """
        Initialize a Sines function space.

        Parameters:
        - N (int): The number of sine basis functions to include in the space.
        - domain (tuple, optional): The domain of the sine function space. Defaults to (0, 1).
        - bc (tuple, optional): Boundary conditions for the sine basis functions. Defaults to (0, 0).

        This constructor initializes a Sines function space with the specified number of basis functions, domain,
        and boundary conditions. It extends the Trigonometric class and sets up the necessary parameters.
        """
        Trigonometric.__init__(self, N, domain=domain)
        self.B = Dirichlet(bc, domain, self.reference_domain)

    def basis_function(self, j, sympy=False):
        """
        Get the j-th sine basis function.

        Parameters:
        - j (int): The index of the sine basis function to retrieve.
        - sympy (bool, optional): If True, return the basis function as a SymPy expression. Defaults to False.

        Returns:
        - function or sympy.Expr: The j-th sine basis function.

        This method retrieves and returns the j-th sine basis function within the sine function space.
        It provides flexibility by allowing you to choose between returning a callable function (default) or
        a symbolic expression (using SymPy) representing the basis function.
        """
        if sympy:
            return sp.sin((j+1)*sp.pi*x)
        return lambda Xj: np.sin((j+1)*np.pi*Xj)

    def derivative_basis_function(self, j, k=1):
        """
        Get the k-th derivative of the j-th sine basis function.

        Parameters:
        - j (int): The index of the sine basis function for which to calculate the derivative.
        - k (int, optional): The order of the derivative to calculate. Defaults to 1.

        Returns:
        - function: The k-th derivative of the j-th sine basis function.

        This method calculates and returns the k-th derivative of the j-th sine basis function within the sine function space.
        It provides a callable function representing the derivative.
        """
        scale = ((j+1)*np.pi)**k * {0: 1, 1: -1}[(k//2) % 2]
        if k % 2 == 0:
            return lambda Xj: scale*np.sin((j+1)*np.pi*Xj)
        else:
            return lambda Xj: scale*np.cos((j+1)*np.pi*Xj)

    def L2_norm_sq(self, N):
        """
        Calculate the square of the L2 norm of sine basis functions.

        Parameters:
        - N (int): The number of sine basis functions to include in the calculation.

        Returns:
        - float: The square of the L2 norm of sine basis functions.

        This method calculates the square of the L2 norm of sine basis functions up to the N-th basis function.
        The L2 norm quantifies the magnitude of these basis functions within the function space.
        """
        return 0.5



class Cosines(Trigonometric):
    """
    Class representing a function space based on cosine basis functions (Not Implemented).

    This class extends the Trigonometric class and is intended to represent a function space based on cosine basis functions.
    Cosine functions are commonly used in various mathematical and numerical applications, including Fourier analysis.

    Attributes (inherited from Trigonometric):
    - N (int): The number of basis functions (Not Implemented).
    - domain (tuple): The domain of the cosine function space (Not Implemented).

    Methods (Not Implemented):
    - basis_function(j, sympy=False): Get the j-th cosine basis function (Not Implemented).
    - derivative_basis_function(j, k=1): Get the k-th derivative of the j-th cosine basis function (Not Implemented).
    - L2_norm_sq(N): Calculate the square of the L2 norm of cosine basis functions (Not Implemented).

    Example:
    - To create a Cosines object (Not Implemented): cosine_space = Cosines(N=10, domain=(0, 2*np.pi))
    """
    
    def __init__(self, N, domain=(0, 1), bc=(0, 0)):
        """
        Initialize a Cosines function space (Not Implemented).

        Parameters (Not Implemented):
        - N (int): The number of cosine basis functions to include in the space (Not Implemented).
        - domain (tuple, optional): The domain of the cosine function space. Defaults to (0, 1) (Not Implemented).
        - bc (tuple, optional): Boundary conditions for the cosine basis functions. Defaults to (0, 0) (Not Implemented).

        This constructor is intended to initialize a Cosines function space with the specified number of basis functions,
        domain, and boundary conditions. (Not Implemented)
        """
        raise NotImplementedError("The Cosines class is not implemented.")

    def basis_function(self, j, sympy=False):
        """
        Get the j-th cosine basis function (Not Implemented).

        Parameters:
        - j (int): The index of the cosine basis function to retrieve (Not Implemented).
        - sympy (bool, optional): If True, return the basis function as a SymPy expression. Defaults to False (Not Implemented).

        Returns (Not Implemented):
        - function or sympy.Expr: The j-th cosine basis function (Not Implemented).

        This method is intended to retrieve and return the j-th cosine basis function within the cosine function space.
        It is expected to provide flexibility by allowing you to choose between returning a callable function (default)
        or a symbolic expression (using SymPy) representing the basis function. (Not Implemented)
        """
        raise NotImplementedError("The Cosines class is not implemented.")

    def derivative_basis_function(self, j, k=1):
        """
        Get the k-th derivative of the j-th cosine basis function (Not Implemented).

        Parameters:
        - j (int): The index of the cosine basis function for which to calculate the derivative (Not Implemented).
        - k (int, optional): The order of the derivative to calculate. Defaults to 1 (Not Implemented).

        Returns (Not Implemented):
        - function: The k-th derivative of the j-th cosine basis function (Not Implemented).

        This method is intended to calculate and return the k-th derivative of the j-th cosine basis function
        within the cosine function space. It should provide a callable function representing the derivative. (Not Implemented)
        """
        raise NotImplementedError("The Cosines class is not implemented.")

    def L2_norm_sq(self, N):
        """
        Calculate the square of the L2 norm of cosine basis functions (Not Implemented).

        Parameters:
        - N (int): The number of cosine basis functions to include in the calculation (Not Implemented).

        Returns (Not Implemented):
        - float: The square of the L2 norm of cosine basis functions (Not Implemented).

        This method is intended to calculate the square of the L2 norm of cosine basis functions up to the N-th basis function.
        The L2 norm quantifies the magnitude of these basis functions within the function space. (Not Implemented)
        """
        raise NotImplementedError("The Cosines class is not implemented.")

class Dirichlet:
    """
    Class representing Dirichlet boundary conditions.

    This class is used to define Dirichlet boundary conditions for a function space within a specified domain.
    Dirichlet boundary conditions prescribe the values of the function at the boundaries of the domain.

    Attributes:
    - bc (tuple): A tuple representing the boundary conditions as (left_value, right_value).
    - x (sympy.Expr): The boundary function in physical coordinates.
    - xX (sympy.Expr): The boundary function in reference coordinates.
    - Xl (function): A function that evaluates the boundary function in reference coordinates.

    Parameters:
    - bc (tuple): A tuple representing the boundary conditions as (left_value, right_value).
    - domain (tuple): The domain of the function space.
    - reference_domain (tuple): The reference domain for the function space.

    Example:
    - To create a Dirichlet object with boundary conditions (0, 1): dirichlet_bc = Dirichlet(bc=(0, 1), domain=(0, 1), reference_domain=(0, 1))
    """

    def __init__(self, bc, domain, reference_domain):
        """
        Initialize a Dirichlet boundary condition.

        Parameters:
        - bc (tuple): A tuple representing the boundary conditions as (left_value, right_value).
        - domain (tuple): The domain of the function space.
        - reference_domain (tuple): The reference domain for the function space.

        This constructor initializes a Dirichlet boundary condition with the specified boundary conditions, domain,
        and reference domain. It calculates the boundary functions in both physical and reference coordinates.
        """
        d = domain
        r = reference_domain
        h = d[1] - d[0]
        self.bc = bc
        self.x = bc[0] * (d[1] - x) / h + bc[1] * (x - d[0]) / h  # in physical coordinates
        self.xX = map_expression_true_domain(self.x, x, d, r)  # in reference coordinates
        self.Xl = sp.lambdify(x, self.xX)

class Neumann:
    """
    Class representing Neumann boundary conditions.

    This class is used to define Neumann boundary conditions for a function space within a specified domain.
    Neumann boundary conditions prescribe the flux or derivative of the function at the boundaries of the domain.

    Attributes:
    - bc (tuple): A tuple representing the boundary conditions as (left_flux, right_flux).
    - x (sympy.Expr): The boundary function in physical coordinates.
    - xX (sympy.Expr): The boundary function in reference coordinates.
    - Xl (function): A function that evaluates the boundary function in reference coordinates.

    Parameters:
    - bc (tuple): A tuple representing the boundary conditions as (left_flux, right_flux).
    - domain (tuple): The domain of the function space.
    - reference_domain (tuple): The reference domain for the function space.

    Example:
    - To create a Neumann object with boundary conditions (1, -1): neumann_bc = Neumann(bc=(1, -1), domain=(0, 1), reference_domain=(0, 1))
    """

    def __init__(self, bc, domain, reference_domain):
        """
        Initialize a Neumann boundary condition.

        Parameters:
        - bc (tuple): A tuple representing the boundary conditions as (left_flux, right_flux).
        - domain (tuple): The domain of the function space.
        - reference_domain (tuple): The reference domain for the function space.

        This constructor initializes a Neumann boundary condition with the specified boundary conditions, domain,
        and reference domain. It calculates the boundary functions in both physical and reference coordinates.
        """
        d = domain
        r = reference_domain
        h = d[1] - d[0]
        self.bc = bc
        self.x = bc[0] / h * (d[1] * x - x**2 / 2) + bc[1] / h * (x**2 / 2 - d[0] * x)  # in physical coordinates
        self.xX = map_expression_true_domain(self.x, x, d, r)  # in reference coordinates
        self.Xl = sp.lambdify(x, self.xX)



class Composite(FunctionSpace):
    """Base class for function spaces created as linear combinations of orthogonal basis functions

    The composite basis functions are defined using the orthogonal basis functions
    (Chebyshev or Legendre) and a stencil matrix S. The stencil matrix S is used
    such that basis function i is

    .. math::

        \psi_i = \sum_{j=0}^N S_{ij} Q_j

    where :math:`Q_i` can be either the i'th Chebyshev or Legendre polynomial

    For example, both Chebyshev and Legendre have Dirichlet basis functions

    .. math::

        \psi_i = Q_i-Q_{i+2}

    Here the stencil matrix will be

    .. math::

        s_{ij} = \delta_{ij} - \delta_{i+2, j}, \quad (i, j) \in (0, 1, \ldots, N) \times (0, 1, \ldots, N+2)

    Note that the stencil matrix is of shape :math:`(N+1) \times (N+3)`.

    Attributes (inherited from FunctionSpace):
    - N (int): The number of basis functions.
    - domain (tuple): The domain of the function space.
    - reference_domain (tuple): The reference domain for the function space.

    Methods:
    - eval(uh, xj): Evaluate the composite basis functions at the specified points.
    - mass_matrix(): Compute the mass matrix of the composite function space.

    Example:
    - To create a Composite function space: composite_space = Composite(N=10, domain=(0, 1), reference_domain=(0, 1))
    """

    def eval(self, uh, xj):
        """
        Evaluate the composite basis functions at the specified points.

        Parameters:
        - uh (array-like): The coefficients of the composite basis functions.
        - xj (array-like): The evaluation points in physical coordinates.

        Returns:
        - array-like: The values of the composite basis functions at the specified points.

        This method calculates the values of the composite basis functions at the given evaluation points.
        """
        xj = np.atleast_1d(xj)
        Xj = map_reference_domain(xj, self.domain, self.reference_domain)
        P = self.eval_basis_function_all(Xj)
        return P @ uh + self.B.Xl(Xj)

    def mass_matrix(self):
        """
        Compute the mass matrix of the composite function space.

        Returns:
        - scipy.sparse.csr_matrix: The mass matrix.

        This method computes the mass matrix for the composite function space. The mass matrix is typically used in
        solving partial differential equations involving these basis functions.
        """
        M = sparse.diags([self.L2_norm_sq(self.N+3)], [0],
                         shape=(self.N+3, self.N+3), format='csr')
        return self.S @ M @ self.S.T


class DirichletLegendre(Composite, Legendre):
    """
    Class representing a function space with Dirichlet boundary conditions based on Legendre polynomials.

    This class extends both the Composite and Legendre classes to create a function space with Dirichlet boundary conditions.
    The basis functions are constructed as a linear combination of Legendre polynomials with a stencil matrix.

    Attributes (inherited from Composite and Legendre):
    - N (int): The number of basis functions.
    - domain (tuple): The domain of the function space.
    - reference_domain (tuple): The reference domain for the function space.

    Methods (inherited from Composite and Legendre):
    - eval(uh, xj): Evaluate the basis functions at specified points.
    - mass_matrix(): Compute the mass matrix of the function space.

    Parameters:
    - N (int): The number of basis functions to include in the space.
    - domain (tuple, optional): The domain of the function space. Defaults to (-1, 1).
    - bc (tuple, optional): Boundary conditions for the Dirichlet basis functions. Defaults to (0, 0).

    Example:
    - To create a DirichletLegendre function space with N=10, domain=(-1, 1), and Dirichlet boundary conditions (0, 0):
      dirichlet_legendre_space = DirichletLegendre(N=10, domain=(-1, 1), bc=(0, 0))
    """

    def __init__(self, N, domain=(-1, 1), bc=(0, 0)):
        """
        Initialize a DirichletLegendre function space.

        Parameters:
        - N (int): The number of basis functions to include in the space.
        - domain (tuple, optional): The domain of the function space. Defaults to (-1, 1).
        - bc (tuple, optional): Boundary conditions for the Dirichlet basis functions. Defaults to (0, 0).

        This constructor initializes a DirichletLegendre function space with the specified number of basis functions,
        domain, and Dirichlet boundary conditions. It also constructs the stencil matrix and initializes the Legendre basis.
        """
        Legendre.__init__(self, N, domain=domain)
        self.B = Dirichlet(bc, domain, self.reference_domain)
        self.S = sparse.diags((1, -1), (0, 2), shape=(N+1, N+3), format='csr')

    def basis_function(self, j, sympy=False):
        """
        Get the j-th basis function of the DirichletLegendre function space.

        Parameters:
        - j (int): The index of the basis function to retrieve.
        - sympy (bool, optional): If True, return the basis function as a SymPy expression. Defaults to False.

        Returns:
        - function or sympy.Expr: The j-th basis function.

        This method retrieves and returns the j-th basis function of the DirichletLegendre function space.
        """
        raise NotImplementedError("The basis_function method is not implemented for DirichletLegendre.")

class NeumannLegendre(Composite, Legendre):
    """
    Class representing a function space with Neumann boundary conditions based on Legendre polynomials.

    This class extends both the Composite and Legendre classes to create a function space with Neumann boundary conditions.
    The basis functions are constructed as a linear combination of Legendre polynomials with a stencil matrix.

    Attributes (inherited from Composite and Legendre):
    - N (int): The number of basis functions.
    - domain (tuple): The domain of the function space.
    - reference_domain (tuple): The reference domain for the function space.

    Methods (inherited from Composite and Legendre):
    - eval(uh, xj): Evaluate the basis functions at specified points.
    - mass_matrix(): Compute the mass matrix of the function space.

    Parameters:
    - N (int): The number of basis functions to include in the space.
    - domain (tuple, optional): The domain of the function space. Defaults to (-1, 1).
    - bc (tuple, optional): Boundary conditions for the Neumann basis functions. Defaults to (0, 0).
    - constraint (int, optional): A constraint parameter (not implemented). Defaults to 0.

    Example:
    - To create a NeumannLegendre function space with N=10, domain=(-1, 1), and Neumann boundary conditions (0, 0):
      neumann_legendre_space = NeumannLegendre(N=10, domain=(-1, 1), bc=(0, 0))
    """

    def __init__(self, N, domain=(-1, 1), bc=(0, 0), constraint=0):
        """
        Initialize a NeumannLegendre function space.

        Parameters:
        - N (int): The number of basis functions to include in the space.
        - domain (tuple, optional): The domain of the function space. Defaults to (-1, 1).
        - bc (tuple, optional): Boundary conditions for the Neumann basis functions. Defaults to (0, 0).
        - constraint (int, optional): A constraint parameter (not implemented). Defaults to 0.

        This constructor initializes a NeumannLegendre function space with the specified number of basis functions,
        domain, Neumann boundary conditions, and constraint (if applicable). It also constructs the stencil matrix and
        initializes the Legendre basis.
        """
        raise NotImplementedError

    def basis_function(self, j, sympy=False):
        """
        Get the j-th basis function of the NeumannLegendre function space.

        Parameters:
        - j (int): The index of the basis function to retrieve.
        - sympy (bool, optional): If True, return the basis function as a SymPy expression. Defaults to False.

        Returns:
        - function or sympy.Expr: The j-th basis function.

        This method retrieves and returns the j-th basis function of the NeumannLegendre function space.
        """
        raise NotImplementedError

class DirichletChebyshev(Composite, Chebyshev):
    """
    Class representing a function space with Dirichlet boundary conditions based on Chebyshev polynomials.

    This class extends both the Composite and Chebyshev classes to create a function space with Dirichlet boundary conditions.
    The basis functions are constructed as a linear combination of Chebyshev polynomials with a stencil matrix.

    Attributes (inherited from Composite and Chebyshev):
    - N (int): The number of basis functions.
    - domain (tuple): The domain of the function space.
    - reference_domain (tuple): The reference domain for the function space.

    Methods (inherited from Composite and Chebyshev):
    - eval(uh, xj): Evaluate the basis functions at specified points.
    - mass_matrix(): Compute the mass matrix of the function space.

    Parameters:
    - N (int): The number of basis functions to include in the space.
    - domain (tuple, optional): The domain of the function space. Defaults to (-1, 1).
    - bc (tuple, optional): Boundary conditions for the Dirichlet basis functions. Defaults to (0, 0).

    Example:
    - To create a DirichletChebyshev function space with N=10, domain=(-1, 1), and Dirichlet boundary conditions (0, 0):
      dirichlet_chebyshev_space = DirichletChebyshev(N=10, domain=(-1, 1), bc=(0, 0))
    """

    def __init__(self, N, domain=(-1, 1), bc=(0, 0)):
        """
        Initialize a DirichletChebyshev function space.

        Parameters:
        - N (int): The number of basis functions to include in the space.
        - domain (tuple, optional): The domain of the function space. Defaults to (-1, 1).
        - bc (tuple, optional): Boundary conditions for the Dirichlet basis functions. Defaults to (0, 0).

        This constructor initializes a DirichletChebyshev function space with the specified number of basis functions,
        domain, and Dirichlet boundary conditions. It also constructs the stencil matrix and initializes the Chebyshev basis.
        """
        Chebyshev.__init__(self, N, domain=domain)
        self.B = Dirichlet(bc, domain, self.reference_domain)
        self.S = sparse.diags((1, -1), (0, 2), shape=(N+1, N+3), format='csr')

    def basis_function(self, j, sympy=False):
        """
        Get the j-th basis function of the DirichletChebyshev function space.

        Parameters:
        - j (int): The index of the basis function to retrieve.
        - sympy (bool, optional): If True, return the basis function as a SymPy expression. Defaults to False.

        Returns:
        - function or sympy.Expr: The j-th basis function.

        This method retrieves and returns the j-th basis function of the DirichletChebyshev function space.
        """
        if sympy:
            return sp.cos(j*sp.acos(x)) - sp.cos((j+2)*sp.acos(x))
        return Cheb.basis(j) - Cheb.basis(j+2)


class NeumannChebyshev(Composite, Chebyshev):
    """
    Class representing a function space with Neumann boundary conditions based on Chebyshev polynomials.

    This class extends both the Composite and Chebyshev classes to create a function space with Neumann boundary conditions.
    The basis functions are constructed as a linear combination of Chebyshev polynomials with a stencil matrix.

    Attributes (inherited from Composite and Chebyshev):
    - N (int): The number of basis functions.
    - domain (tuple): The domain of the function space.
    - reference_domain (tuple): The reference domain for the function space.

    Methods (inherited from Composite and Chebyshev):
    - eval(uh, xj): Evaluate the basis functions at specified points.
    - mass_matrix(): Compute the mass matrix of the function space.

    Parameters:
    - N (int): The number of basis functions to include in the space.
    - domain (tuple, optional): The domain of the function space. Defaults to (-1, 1).
    - bc (tuple, optional): Boundary conditions for the Neumann basis functions. Defaults to (0, 0).
    - constraint (int, optional): A constraint parameter (not implemented). Defaults to 0.

    Example:
    - To create a NeumannChebyshev function space with N=10, domain=(-1, 1), and Neumann boundary conditions (0, 0):
      neumann_chebyshev_space = NeumannChebyshev(N=10, domain=(-1, 1), bc=(0, 0))
    """

    def __init__(self, N, domain=(-1, 1), bc=(0, 0), constraint=0):
        """
        Initialize a NeumannChebyshev function space.

        Parameters:
        - N (int): The number of basis functions to include in the space.
        - domain (tuple, optional): The domain of the function space. Defaults to (-1, 1).
        - bc (tuple, optional): Boundary conditions for the Neumann basis functions. Defaults to (0, 0).
        - constraint (int, optional): A constraint parameter (not implemented). Defaults to 0.

        This constructor initializes a NeumannChebyshev function space with the specified number of basis functions,
        domain, Neumann boundary conditions, and constraint (if applicable). It also constructs the stencil matrix and
        initializes the Chebyshev basis.
        """
        raise NotImplementedError

    def basis_function(self, j, sympy=False):
        """
        Get the j-th basis function of the NeumannChebyshev function space.

        Parameters:
        - j (int): The index of the basis function to retrieve.
        - sympy (bool, optional): If True, return the basis function as a SymPy expression. Defaults to False.

        Returns:
        - function or sympy.Expr: The j-th basis function.

        This method retrieves and returns the j-th basis function of the NeumannChebyshev function space.
        """
        raise NotImplementedError


class BasisFunction:
    """
    Class representing a basis function for a function space.

    This class allows you to define a basis function with a specified number of derivatives and an argument.

    Attributes:
    - function_space (FunctionSpace): The function space associated with the basis function.

    Parameters:
    - V (FunctionSpace): The function space to which the basis function belongs.
    - diff (int, optional): The number of derivatives applied to the basis function. Defaults to 0.
    - argument (int, optional): An argument associated with the basis function. Defaults to 0.

    Properties:
    - argument: The argument associated with the basis function.
    - function_space: The function space to which the basis function belongs.
    - num_derivatives: The number of derivatives applied to the basis function.

    Methods:
    - diff(k): Create a new basis function with additional derivatives.

    Example:
    - To create a BasisFunction with 2 derivatives, associated with a function space V:
      basis = BasisFunction(V, diff=2)
    """

    def __init__(self, V, diff=0, argument=0):
        """
        Initialize a BasisFunction.

        Parameters:
        - V (FunctionSpace): The function space to which the basis function belongs.
        - diff (int, optional): The number of derivatives applied to the basis function. Defaults to 0.
        - argument (int, optional): An argument associated with the basis function. Defaults to 0.

        This constructor initializes a BasisFunction with the specified function space, number of derivatives,
        and argument.
        """
        self._V = V
        self._num_derivatives = diff
        self._argument = argument

    @property
    def argument(self):
        """
        Get the argument associated with the basis function.

        Returns:
        - int: The argument value.

        This property allows you to retrieve the argument associated with the basis function.
        """
        return self._argument

    @property
    def function_space(self):
        """
        Get the function space to which the basis function belongs.

        Returns:
        - FunctionSpace: The associated function space.

        This property allows you to retrieve the function space associated with the basis function.
        """
        return self._V

    @property
    def num_derivatives(self):
        """
        Get the number of derivatives applied to the basis function.

        Returns:
        - int: The number of derivatives.

        This property allows you to retrieve the number of derivatives applied to the basis function.
        """
        return self._num_derivatives

    def diff(self, k):
        """
        Create a new basis function with additional derivatives.

        Parameters:
        - k (int): The number of additional derivatives to apply.

        Returns:
        - BasisFunction: A new basis function with the specified number of additional derivatives.

        This method creates a new BasisFunction by applying additional derivatives to the current basis function.
        """
        return self.__class__(self.function_space, diff=self.num_derivatives + k)


class TestFunction(BasisFunction):
    """
    Class representing a test function in a finite element framework.

    This class extends the BasisFunction class to represent a test function used in a finite element framework.
    It inherits the attributes and methods from BasisFunction.

    Parameters:
    - V (FunctionSpace): The function space to which the test function belongs.
    - diff (int, optional): The number of derivatives applied to the test function. Defaults to 0.

    Example:
    - To create a TestFunction associated with a function space V with 2 derivatives:
      test_function = TestFunction(V, diff=2)
    """

    def __init__(self, V, diff=0):
        """
        Initialize a TestFunction.

        Parameters:
        - V (FunctionSpace): The function space to which the test function belongs.
        - diff (int, optional): The number of derivatives applied to the test function. Defaults to 0.

        This constructor initializes a TestFunction with the specified function space and number of derivatives.
        By default, the argument is set to 0 to indicate that it is a test function.
        """
        BasisFunction.__init__(self, V, diff=diff, argument=0)


class TrialFunction(BasisFunction):
    """
    Class representing a trial function in a finite element framework.

    This class extends the BasisFunction class to represent a trial function used in a finite element framework.
    It inherits the attributes and methods from BasisFunction.

    Parameters:
    - V (FunctionSpace): The function space to which the trial function belongs.
    - diff (int, optional): The number of derivatives applied to the trial function. Defaults to 0.

    Example:
    - To create a TrialFunction associated with a function space V with 1 derivative:
      trial_function = TrialFunction(V, diff=1)
    """

    def __init__(self, V, diff=0):
        """
        Initialize a TrialFunction.

        Parameters:
        - V (FunctionSpace): The function space to which the trial function belongs.
        - diff (int, optional): The number of derivatives applied to the trial function. Defaults to 0.

        This constructor initializes a TrialFunction with the specified function space and number of derivatives.
        By default, the argument is set to 1 to indicate that it is a trial function.
        """
        BasisFunction.__init__(self, V, diff=diff, argument=1)


def assemble_generic_matrix(u, v):
    """
    Assemble a generic matrix for finite element computations.

    This function assembles a generic matrix for finite element computations, given a trial function `u` and a test
    function `v`. The resulting matrix represents the inner product of the two functions.

    Parameters:
    - u (TrialFunction): The trial function used in the computation.
    - v (TestFunction): The test function used in the computation.

    Returns:
    - np.ndarray: The assembled generic matrix as a NumPy array.

    This function performs the assembly of the generic matrix for finite element computations, taking into account
    the trial and test functions provided. It calculates the inner product of these functions and returns the
    resulting matrix.

    Note:
    - The trial function `u` and test function `v` must belong to the same function space.
    - The matrix assembly is done using numerical quadrature methods.

    Example:
    - To assemble a generic matrix for a given trial function `u` and test function `v`, use:
      matrix = assemble_generic_matrix(u, v)
    """
    assert isinstance(u, TrialFunction)
    assert isinstance(v, TestFunction)
    V = v.function_space
    assert u.function_space == V
    r = V.reference_domain
    D = np.zeros((V.N + 1, V.N + 1))
    cheb = V.weight() == 1 / sp.sqrt(1 - x ** 2)
    symmetric = True if u.num_derivatives == v.num_derivatives else False
    w = {'weight': 'alg' if cheb else None,
         'wvar': (-0.5, -0.5) if cheb else None}

    def uv(Xj, i, j):
        return (V.evaluate_derivative_basis_function(Xj, i, k=v.num_derivatives) *
                V.evaluate_derivative_basis_function(Xj, j, k=u.num_derivatives))

    for i in range(V.N + 1):
        for j in range(i if symmetric else 0, V.N + 1):
            D[i, j] = quad(uv, float(r[0]), float(r[1]), args=(i, j), **w)[0]
            if symmetric:
                D[j, i] = D[i, j]

    return D


def inner(u, v: TestFunction):
    """
    Calculate the inner product of functions in a finite element framework.

    This function calculates the inner product of two functions, where one function (`u`) belongs to the trial function
    space and the other (`v`) belongs to the test function space within a finite element framework.

    Parameters:
    - u: A function from the trial function space.
    - v (TestFunction): A test function used in the computation.

    Returns:
    - np.ndarray or float: The inner product result, either as a NumPy array (for non-scalar cases) or a float.

    This function determines the appropriate method to calculate the inner product based on the provided trial function
    `u` and test function `v`. It considers various cases, including the number of derivatives and whether the inner
    product can be calculated directly from the mass matrix or requires generic matrix assembly.

    Notes:
    - The trial function `u` and test function `v` must belong to the same function space.
    - The function chooses the most efficient method to calculate the inner product based on the provided functions.

    Example:
    - To calculate the inner product of a trial function `u` and a test function `v`, use:
      result = inner(u, v)
    """
    V = v.function_space
    h = V.domain_factor
    if isinstance(u, TrialFunction):
        num_derivatives = u.num_derivatives + v.num_derivatives
        if num_derivatives == 0:
            return float(h) * V.mass_matrix()
        else:
            return float(h) ** (1 - num_derivatives) * assemble_generic_matrix(u, v)
    return V.inner_product(u)


def project(ue, V):
    """
    Project a function onto a finite element function space.

    This function projects a given function `ue` onto a finite element function space `V`. It calculates the projection
    by solving a linear system using the inner products of the trial and test functions.

    Parameters:
    - ue: The function to be projected onto the function space.
    - V: The finite element function space onto which `ue` is projected.

    Returns:
    - np.ndarray: The projected function as a NumPy array.

    This function performs the projection of a function onto a finite element function space. It constructs the linear
    system based on the inner products of trial and test functions and then solves it to obtain the projected function.

    Example:
    - To project a function `ue` onto a finite element function space `V`, use:
      projected_function = project(ue, V)
    """
    u = TrialFunction(V)
    v = TestFunction(V)
    b = inner(ue, v)
    A = inner(u, v)
    uh = sparse.linalg.spsolve(A, b)
    return uh




# 
# 
# 
# 
# 
# 
# 
# remember to adress undefined uj here.
# 
# 
# 
# 
# 
# 
# 
# 

def L2_error(uh, ue, V, kind='norm'):
    """
    Calculate the L2 error of a finite element solution.

    This function calculates the L2 error between a finite element solution `uh` and an exact solution `ue` within the
    context of a finite element function space `V`.

    Parameters:
    - uh: The finite element solution to be evaluated.
    - ue: The exact solution for comparison.
    - V: The finite element function space in which the solutions reside.
    - kind (str): The type of error calculation to perform, either 'norm' (default) for L2 norm or 'inf' for infinity
      norm.

    Returns:
    - float: The L2 error between `uh` and `ue` if `kind` is 'norm', or the maximum pointwise error if `kind` is 'inf'.

    This function calculates the L2 error between a finite element solution and an exact solution within a given function
    space. The error can be calculated as the L2 norm (Euclidean norm) of the difference between the solutions or as
    the maximum pointwise error, depending on the specified `kind`.

    Note:
    - The `kind` parameter determines the type of error calculation to perform. Use 'norm' for L2 norm error (default)
      or 'inf' for maximum pointwise error.
    - The function space `V` should be consistent with the spaces of `uh` and `ue`.

    Example:
    - To calculate the L2 error between a finite element solution `uh` and an exact solution `ue`, use:
      error = L2_error(uh, ue, V, kind='norm')
    """
    d = V.domain
    uej = sp.lambdify(x, ue)

    def uv(xj):
        return (uej(xj) - V.eval(uh, xj)) ** 2

    if kind == 'norm':
        return np.sqrt(quad(uv, float(d[0]), float(d[1]))[0])
    elif kind == 'inf':
        return max(abs(uj - uej))


def test_project():
    """
    Test the projection of a known exact solution onto finite element function spaces.

    This function performs the following steps:
    1. Defines an exact solution `ue` using a special function (Bessel function in this case).
    2. Specifies the domain over which the projection will be performed.
    3. Iterates over two types of finite element function spaces: Chebyshev and Legendre.
    4. For each function space, it creates the function space `V` with a specific number of basis functions.
    5. Projects the exact solution `ue` onto the function space `V` using the `project` function.
    6. Calculates the L2 error between the projected solution `u` and the exact solution `ue` using the `L2_error` function.
    7. Prints the L2 error, the number of basis functions (N), and the class name of the function space.
    8. Asserts that the L2 error is below a predefined threshold (1e-6).

    This test ensures the accuracy of the projection onto finite element spaces for different function spaces.

    Raises:
        AssertionError: If the L2 error exceeds the predefined threshold for any function space.

    Note:
        - The `project` and `L2_error` functions are used to perform the projection and compute the error.
        - The exact solution `ue` is defined using a Bessel function.

    """
    ue = sp.besselj(0, x)  # Define the exact solution using a Bessel function
    domain = (0, 10)  # Specify the domain over which the projection will be performed

    # Iterate over two types of finite element function spaces: Chebyshev and Legendre
    for space in (Chebyshev, Legendre):
        V = space(16, domain=domain)  # Create the function space with 16 basis functions
        u = project(ue, V)  # Project the exact solution onto the function space
        err = L2_error(u, ue, V)  # Calculate the L2 error between projected and exact solutions

        # Print the results, including L2 error, number of basis functions (N), and function space class name
        print(
            f'test_project: L2 error = {err:2.4e}, N = {V.N}, {V.__class__.__name__}')

        # Assert that the L2 error is below the predefined threshold (1e-6)
        assert err < 1e-6



def test_helmholtz():
    """
    Test the Helmholtz equation solver for different function spaces.

    This function tests the Helmholtz equation solver for various finite element function spaces, including Neumann and Dirichlet boundary conditions,
    Chebyshev, Legendre, Sines, and Cosines function spaces. It computes and compares the L2 error between the computed solution and the exact solution
    for each function space.

    Raises:
    AssertionError: If the computed L2 error exceeds the tolerance (1e-3) for any function space.

    """
    ue = sp.besselj(0, x)  # Exact solution
    f = ue.diff(x, 2) + ue  # Right-hand side of the Helmholtz equation
    domain = (0, 10)  # Domain of the problem

    # Loop through different function spaces and test the solver
    for space in (NeumannChebyshev, NeumannLegendre, DirichletChebyshev, DirichletLegendre, Sines, Cosines):
        if space in (NeumannChebyshev, NeumannLegendre, Cosines):
            # Set boundary conditions for Neumann or Cosines spaces
            bc = ue.diff(x, 1).subs(x, domain[0]), ue.diff(x, 1).subs(x, domain[1])
        else:
            # Set boundary conditions for Dirichlet spaces
            bc = ue.subs(x, domain[0]), ue.subs(x, domain[1])

        # Choose the number of basis functions (N) based on the function space
        N = 60 if space in (Sines, Cosines) else 12

        # Create the function space
        V = space(N, domain=domain, bc=bc)

        # Define trial and test functions
        u = TrialFunction(V)
        v = TestFunction(V)

        # Define the Helmholtz equation and right-hand side
        A = inner(u.diff(2), v) + inner(u, v)
        b = inner(f - (V.B.x.diff(x, 2) + V.B.x), v)

        # Solve for the solution u_tilde
        u_tilde = np.linalg.solve(A, b)

        # Compute the L2 error between u_tilde and the exact solution ue
        err = L2_error(u_tilde, ue, V)

        # Print and assert the L2 error
        print(f'test_helmholtz: L2 error = {err:2.4e}, N = {N}, {V.__class__.__name__}')
        assert err < 1e-3



def test_convection_diffusion():
    """
    Test function for solving the convection-diffusion problem using different function spaces.

    This function performs tests for solving the convection-diffusion problem with various function spaces,
    including DirichletLegendre, DirichletChebyshev, and Sines. It computes the numerical solution and
    compares it with the exact solution, evaluating the L2 error.

    Raises:
        AssertionError: If the computed L2 error exceeds the tolerance threshold.

    Usage:
        Call this function to perform tests for the convection-diffusion problem using different function spaces.
    """
    eps = 0.05
    ue = (sp.exp(-x/eps)-1)/(sp.exp(-1/eps)-1)
    f = 0
    domain = (0, 1)
    for space in (DirichletLegendre, DirichletChebyshev, Sines):
        N = 50 if space is Sines else 16
        V = space(N, domain=domain, bc=(0, 1))
        u = TrialFunction(V)
        v = TestFunction(V)
        A = inner(u.diff(2), v) + (1/eps)*inner(u.diff(1), v)
        b = inner(f-((1/eps)*V.B.x.diff(x, 1)), v)
        u_tilde = np.linalg.solve(A, b)
        err = L2_error(u_tilde, ue, V)
        print(
            f'test_convection_diffusion: L2 error = {err:2.4e}, N = {N}, {V.__class__.__name__}')
        assert err < 1e-3



if __name__ == '__main__':
    test_project()
    test_convection_diffusion()
    test_helmholtz()
