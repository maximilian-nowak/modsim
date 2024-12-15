function [dydt] = N_body_ode(t,y,m)
%N_body_ode Returns the ode to a N-body system.
%
%   In order to pass this function to a ode solver with parameter m, 
%   define an anonymous function which in turn returns this function, e.g.
%
%       m = [1,1,1];
%       [t,y] = ode45(@(t, y)N_body_ode(t,y,m), t_span, y0);
%

    G = 1;  % simplified gravitational constant
    
    N = length(y);
    dydt = zeros(N,1);
    r3_cache = zeros(N,N);  % cache for distances between bodies
    
    % set step size to 6 as y holds each bodies coords and velocities
    for i=1:6:N
        ix = i;
        iy = i+2;
        iz = i+4;
        
        x_inner_sum = 0;
        y_inner_sum = 0;
        z_inner_sum = 0;
        
        for j=1:6:N
            if i ~= j
                jx = j;
                jy = j+2;
                jz = j+4;

                m_j = m((j-1)/6 + 1);
                if ~r3_cache(i, j)
                    r3_cache(i, j) = sqrt( (y(ix)-y(jx))^2 + (y(iy)-y(jy))^2 + (y(iz)-y(jz))^2 )^3;
                    r3_cache(j, i) = r3_cache(i, j);
                end
                
                x_inner_sum = x_inner_sum + m_j*(y(jx)-y(ix))/r3_cache(i, j);
                y_inner_sum = y_inner_sum + m_j*(y(jy)-y(iy))/r3_cache(i, j);
                z_inner_sum = z_inner_sum + m_j*(y(jz)-y(iz))/r3_cache(i, j);
            end
        end
        
        % shifting results to 1st order by setting y = y', and y'= y''
        dydt(ix) = y(ix+1); dydt(ix+1) = G * x_inner_sum;
        dydt(iy) = y(iy+1); dydt(iy+1) = G * y_inner_sum;
        dydt(iz) = y(iz+1); dydt(iz+1) = G * z_inner_sum;
    end
end

