import numpy as np
import pytest
from scipy.integrate import quad_vec
from scipy.linalg import expm
from src.models.velocity_bias import bias_ode,bias_jacobian,frozen_bias_diffusion_covariance,frozen_position_observability


@pytest.mark.parametrize('theta', [0.,.7,-2.5])
def test_bias_sign_and_jacobian(theta):
    x=np.array([0.,0.,theta,-.2]);u=np.array([.8,0.])
    np.testing.assert_allclose(bias_ode(x,u)[:2],[np.cos(theta),np.sin(theta)])
    eps=1e-6;eye=np.eye(4)
    fd=np.column_stack([(bias_ode(x+eps*e,u)-bias_ode(x-eps*e,u))/(2*eps) for e in eye])
    np.testing.assert_allclose(bias_jacobian(x,u),fd,atol=2e-10)


def test_local_observability_moving_and_stationary():
    for theta in [0.,.7,-2.5]:
        x=[0.,0.,theta,0.]
        assert np.linalg.matrix_rank(frozen_position_observability(x,[1.,0.]))==4
        assert np.linalg.matrix_rank(frozen_position_observability(x,[0.,0.]))==3


@pytest.mark.parametrize('dt',[.01,.1,2.])
def test_bias_diffusion_matches_independent_quadrature(dt):
    x=[0.,0.,.7,-.2];u=[.8,0.];density=.002
    A=bias_jacobian(x,u);g=np.array([0.,0.,0.,1.])
    expected=quad_vec(lambda t:density*np.outer(expm(A*t)@g,expm(A*t)@g),0,dt)[0]
    actual=frozen_bias_diffusion_covariance(x,u,dt,density)
    np.testing.assert_allclose(actual,expected,atol=1e-14)
    assert np.linalg.eigvalsh(actual).min()>=-1e-14
    direction=np.array([np.cos(.7),np.sin(.7)])
    assert direction@actual[:2,:2]@direction==pytest.approx(density*dt**3/3)


def test_zero_density_and_invalid_parameters():
    np.testing.assert_array_equal(frozen_bias_diffusion_covariance([0,0,0,0],[1,0],.1,0),np.zeros((4,4)))
    with pytest.raises(ValueError):frozen_bias_diffusion_covariance([0,0,0,0],[1,0],0,1)
    with pytest.raises(ValueError):bias_ode([0,0,0],[1,0])
