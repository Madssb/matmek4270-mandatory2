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

        Raises:
        - RuntimeError: Always raises a RuntimeError.

        This property is intended to represent the reference domain of the FunctionSpace.
        However, it always raises a RuntimeError, indicating that it is not implemented or defined.
        """

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
                # Return a symbolic expression for the basis function (if implemented using SymPy).
                # You may need to implement the symbolic expression separately.
                return NotImplemented  # Placeholder for symbolic expression
        else:
            # Return a placeholder indicating that the basis function is not implemented.
            return NotImplemented  # Placeholder for non-implemented basis function

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
        raise RuntimeError

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
        Get the j-th Legendre polynomial basis function.

        Parameters:
        - j (int): The index of the Legendre polynomial basis function to retrieve.
        - sympy (bool, optional): If True, return a symbolic expression using SymPy.
                                If False (default), return a callable function representing the basis function.

        Returns:
        - function or sympy.Expr: The j-th Legendre polynomial basis function.

        This method is used to retrieve the j-th Legendre polynomial basis function within the Legendre polynomial space.
        By default, it returns a callable function representing the basis function. If 'sympy' is set to True, it returns
        a symbolic expression (using SymPy) representing the basis function.
        """
        if sympy:
            # Return a symbolic expression for the Legendre polynomial basis function (if implemented using SymPy).
            return sp.legendre(j, x)
        else:
            # Return a callable function representing the Legendre polynomial basis function.
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
        raise NotImplementedError("L2_norm_sq is not implemented.")


    def mass_matrix(self):
        """
        Calculate the mass matrix of the Legendre polynomial basis functions.

        Returns:
        - numpy.ndarray: A 2D array representing the mass matrix.

        This method calculates and returns the mass matrix of the Legendre polynomial basis functions within the function space.
        The mass matrix represents the influence of these basis functions on the system's behavior and is commonly used
        in numerical simulations and finite element analysis for solving partial differential equations and other tasks.
        """
        raise NotImplementedError("mass_matrix is not implemented.")


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

    def __init__(self, N, domain=(-1, 1)):
        FunctionSpace.__init__(self, N, domain=domain)

    def basis_function(self, j, sympy=False):
        if sympy:
            return sp.cos(j*sp.acos(x))
        return Cheb.basis(j)

    def derivative_basis_function(self, j, k=1):
        return self.basis_function(j).deriv(k)

    def weight(self, x=x):
        return 1/sp.sqrt(1-x**2)

    def L2_norm_sq(self, N):
        raise NotImplementedError

    def mass_matrix(self):
        raise NotImplementedError

    def eval(self, uh, xj):
        xj = np.atleast_1d(xj)
        Xj = map_reference_domain(xj, self.domain, self.reference_domain)
        return np.polynomial.chebyshev.chebval(Xj, uh)

    def inner_product(self, u):
        us = map_expression_true_domain(
            u, x, self.domain, self.reference_domain)
        # change of variables to x=cos(theta)
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
    """Base class for trigonometric function spaces"""

    @property
    def reference_domain(self):
        return (0, 1)

    def mass_matrix(self):
        return sparse.diags([self.L2_norm_sq(self.N+1)], [0], (self.N+1, self.N+1), format='csr')

    def eval(self, uh, xj):
        xj = np.atleast_1d(xj)
        Xj = map_reference_domain(xj, self.domain, self.reference_domain)
        P = self.eval_basis_function_all(Xj)
        return P @ uh + self.B.Xl(Xj)


class Sines(Trigonometric):

    def __init__(self, N, domain=(0, 1), bc=(0, 0)):
        Trigonometric.__init__(self, N, domain=domain)
        self.B = Dirichlet(bc, domain, self.reference_domain)

    def basis_function(self, j, sympy=False):
        if sympy:
            return sp.sin((j+1)*sp.pi*x)
        return lambda Xj: np.sin((j+1)*np.pi*Xj)

    def derivative_basis_function(self, j, k=1):
        scale = ((j+1)*np.pi)**k * {0: 1, 1: -1}[(k//2) % 2]
        if k % 2 == 0:
            return lambda Xj: scale*np.sin((j+1)*np.pi*Xj)
        else:
            return lambda Xj: scale*np.cos((j+1)*np.pi*Xj)

    def L2_norm_sq(self, N):
        return 0.5


class Cosines(Trigonometric):

    def __init__(self, N, domain=(0, 1), bc=(0, 0)):
        raise NotImplementedError

    def basis_function(self, j, sympy=False):
        raise NotImplementedError

    def derivative_basis_function(self, j, k=1):
        raise NotImplementedError

    def L2_norm_sq(self, N):
        raise NotImplementedError

# Create classes to hold the boundary function

class Dirichlet:

    def __init__(self, bc, domain, reference_domain):
        d = domain
        r = reference_domain
        h = d[1]-d[0]
        self.bc = bc
        self.x = bc[0]*(d[1]-x)/h + bc[1]*(x-d[0])/h           # in physical coordinates
        self.xX = map_expression_true_domain(self.x, x, d, r)  # in reference coordinates
        self.Xl = sp.lambdify(x, self.xX)


class Neumann:

    def __init__(self, bc, domain, reference_domain):
        d = domain
        r = reference_domain
        h = d[1]-d[0]
        self.bc = bc
        self.x = bc[0]/h*(d[1]*x-x**2/2) + bc[1]/h*(x**2/2-d[0]*x)  # in physical coordinates
        self.xX = map_expression_true_domain(self.x, x, d, r)       # in reference coordinates
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
    """

    def eval(self, uh, xj):
        xj = np.atleast_1d(xj)
        Xj = map_reference_domain(xj, self.domain, self.reference_domain)
        P = self.eval_basis_function_all(Xj)
        return P @ uh + self.B.Xl(Xj)

    def mass_matrix(self):
        M = sparse.diags([self.L2_norm_sq(self.N+3)], [0],
                         shape=(self.N+3, self.N+3), format='csr')
        return self.S @ M @ self.S.T


class DirichletLegendre(Composite, Legendre):
    def __init__(self, N, domain=(-1, 1), bc=(0, 0)):
        Legendre.__init__(self, N, domain=domain)
        self.B = Dirichlet(bc, domain, self.reference_domain)
        self.S = sparse.diags((1, -1), (0, 2), shape=(N+1, N+3), format='csr')

    def basis_function(self, j, sympy=False):
        raise NotImplementedError


class NeumannLegendre(Composite, Legendre):
    def __init__(self, N, domain=(-1, 1), bc=(0, 0), constraint=0):
        raise NotImplementedError

    def basis_function(self, j, sympy=False):
        raise NotImplementedError


class DirichletChebyshev(Composite, Chebyshev):

    def __init__(self, N, domain=(-1, 1), bc=(0, 0)):
        Chebyshev.__init__(self, N, domain=domain)
        self.B = Dirichlet(bc, domain, self.reference_domain)
        self.S = sparse.diags((1, -1), (0, 2), shape=(N+1, N+3), format='csr')

    def basis_function(self, j, sympy=False):
        if sympy:
            return sp.cos(j*sp.acos(x)) - sp.cos((j+2)*sp.acos(x))
        return Cheb.basis(j)-Cheb.basis(j+2)


class NeumannChebyshev(Composite, Chebyshev):
    def __init__(self, N, domain=(-1, 1), bc=(0, 0), constraint=0):
        raise NotImplementedError

    def basis_function(self, j, sympy=False):
        raise NotImplementedError


class BasisFunction:

    def __init__(self, V, diff=0, argument=0):
        self._V = V
        self._num_derivatives = diff
        self._argument = argument

    @property
    def argument(self):
        return self._argument

    @property
    def function_space(self):
        return self._V

    @property
    def num_derivatives(self):
        return self._num_derivatives

    def diff(self, k):
        return self.__class__(self.function_space, diff=self.num_derivatives+k)


class TestFunction(BasisFunction):

    def __init__(self, V, diff=0):
        BasisFunction.__init__(self, V, diff=diff, argument=0)


class TrialFunction(BasisFunction):

    def __init__(self, V, diff=0):
        BasisFunction.__init__(self, V, diff=diff, argument=1)


def assemble_generic_matrix(u, v):
    assert isinstance(u, TrialFunction)
    assert isinstance(v, TestFunction)
    V = v.function_space
    assert u.function_space == V
    r = V.reference_domain
    D = np.zeros((V.N+1, V.N+1))
    cheb = V.weight() == 1/sp.sqrt(1-x**2)
    symmetric = True if u.num_derivatives == v.num_derivatives else False
    w = {'weight': 'alg' if cheb else None,
         'wvar': (-0.5, -0.5) if cheb else None}
    def uv(Xj, i, j): return (V.evaluate_derivative_basis_function(Xj, i, k=v.num_derivatives) *
                              V.evaluate_derivative_basis_function(Xj, j, k=u.num_derivatives))
    for i in range(V.N+1):
        for j in range(i if symmetric else 0, V.N+1):
            D[i, j] = quad(uv, float(r[0]), float(r[1]), args=(i, j), **w)[0]
            if symmetric:
                D[j, i] = D[i, j]
    return D


def inner(u, v: TestFunction):
    V = v.function_space
    h = V.domain_factor
    if isinstance(u, TrialFunction):
        num_derivatives = u.num_derivatives + v.num_derivatives
        if num_derivatives == 0:
            return float(h) * V.mass_matrix()
        else:
            return float(h)**(1-num_derivatives) * assemble_generic_matrix(u, v)
    return V.inner_product(u)


def project(ue, V):
    u = TrialFunction(V)
    v = TestFunction(V)
    b = inner(ue, v)
    A = inner(u, v)
    uh = sparse.linalg.spsolve(A, b)
    return uh


def L2_error(uh, ue, V, kind='norm'):
    d = V.domain
    uej = sp.lambdify(x, ue)
    def uv(xj): return (uej(xj)-V.eval(uh, xj))**2
    if kind == 'norm':
        return np.sqrt(quad(uv, float(d[0]), float(d[1]))[0])
    elif kind == 'inf':
        return max(abs(uj-uej))


def test_project():
    ue = sp.besselj(0, x)
    domain = (0, 10)
    for space in (Chebyshev, Legendre):
        V = space(16, domain=domain)
        u = project(ue, V)
        err = L2_error(u, ue, V)
        print(
            f'test_project: L2 error = {err:2.4e}, N = {V.N}, {V.__class__.__name__}')
        assert err < 1e-6


def test_helmholtz():
    ue = sp.besselj(0, x)
    f = ue.diff(x, 2)+ue
    domain = (0, 10)
    for space in (NeumannChebyshev, NeumannLegendre, DirichletChebyshev, DirichletLegendre, Sines, Cosines):
        if space in (NeumannChebyshev, NeumannLegendre, Cosines):
            bc = ue.diff(x, 1).subs(x, domain[0]), ue.diff(
                x, 1).subs(x, domain[1])
        else:
            bc = ue.subs(x, domain[0]), ue.subs(x, domain[1])
        N = 60 if space in (Sines, Cosines) else 12
        V = space(N, domain=domain, bc=bc)
        u = TrialFunction(V)
        v = TestFunction(V)
        A = inner(u.diff(2), v) + inner(u, v)
        b = inner(f-(V.B.x.diff(x, 2)+V.B.x), v)
        u_tilde = np.linalg.solve(A, b)
        err = L2_error(u_tilde, ue, V)
        print(
            f'test_helmholtz: L2 error = {err:2.4e}, N = {N}, {V.__class__.__name__}')
        assert err < 1e-3


def test_convection_diffusion():
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
