% Implementation of task 2
% This script simulates the DART mission in a simplified context.
% ---------------------------------------------------

clear all;
close all;

m_sat = 0.00000000006743;
m_dimorphos = 88;
m_didymos = 850;
m = [m_sat, m_didymos, m_dimorphos]; % simplified mass    


% define global to ensure visibility in minimizer function
global dart_ode;
dart_ode = @(t, y)N_body_ode(t,y,m);

%start velocity
x0 = 5;
%velocity bounds without physical correlation
lower_bounds = [0];
upper_bounds = [20];

%minimum values which are returned by the minimize function
%multiple runs with different start-values to get global minimum
real_min = 100;
real_min_x = 0;

%max time where collision has to be reached
t_max = 140;
distance_goal_hit = 0.01;
distance_fail_hit = 0.1;

global t_span;
t_span = [0 200];

global y0;
options = odeset('RelTol',1e-5,'AbsTol',1e-7);

y0 = [
        0; 0.5774;
        0; 0.5774;
        0; 0.5774;
        
        100; 1;
        100; 0;
        100; 0;
        
        105; 1;
        105; 9;
        105; 0;
    ];



for k=[x0:2:80]
    [x,fval,exitflag,output] = fminsearchbnd(@to_minimize, [k], lower_bounds, upper_bounds);
    if (fval < real_min)
        %apply calculated velocity to start-vector
        y_tmp = y0;
        y_tmp(2) = y_tmp(2)*x;
        y_tmp(4) = y_tmp(4)*x;
        y_tmp(6) = y_tmp(6)*x;
        %simulate
        [t,y] = ode45(dart_ode, t_span, y_tmp, options);
        distances = calc_distances(y);
        index_goal_hit = find(distances(:,1)<=distance_goal_hit);
        index_fail_hit = find(distances(:,2)<=distance_fail_hit);
        %check if satellit hit dimorphos before it hits didymos
        if (any(min(index_goal_hit) < min(index_fail_hit)) || (~isempty(min(index_goal_hit)) && isempty(min(index_fail_hit))) )
            % check if hit is before given maximum time
            if (t(min(index_goal_hit)) <= t_max)
                disp("Found valid hit")
                real_min = fval;
                real_min_x = x;
                break;
            end
            disp("Found hit after max time")
        end
    end
end


%Calculate Simulation of valid hit
y_tmp = y0;
y_tmp(2) = y_tmp(2)*real_min_x;
y_tmp(4) = y_tmp(2)*real_min_x;
y_tmp(6) = y_tmp(2)*real_min_x;

[t,y] = ode45(dart_ode, t_span, y_tmp, options);

distances = calc_distances(y);
min_distance = min(distances(:,1));
index_min_abstand = find(distances(:,1) == min_distance);

x1 = y(:, 1);
y1 = y(:, 3);
z1 = y(:, 5);

x2 = y(:, 7);
y2 = y(:, 9);
z2 = y(:, 11);

x3 = y(:, 13);
y3 = y(:, 15);
z3 = y(:, 17);



%configuration for animation
body1 = plot3(x1, y1, z1, 'r.', 'MarkerSize', 20)
hold on
body2 = plot3(x2, y2, z2, 'g.', 'MarkerSize', 20)
body3 = plot3(x3, y3, z3, 'b.', 'MarkerSize', 20)
h1 = animatedline('Color','r', 'MaximumNumPoints', 400, 'LineWidth',3);
h2 = animatedline('Color','g', 'MaximumNumPoints', 400, 'LineWidth',3);
h3 = animatedline('Color','b', 'MaximumNumPoints', 400, 'LineWidth',3);

grid on
xlabel('X');
ylabel('Y');
zlabel('Z');
%axis([0 10 0 10 0 5])
legend('Satellit','Didymos', 'Dimorphos','Satellit','Didymos', 'Dimorphos')

for n=index_min_abstand-1000:1:length(y(:, 1))
    %calc new coordinate origin
    min_x = min([x1(n), x2(n), x3(n)])-20;
    min_y = min([y1(n), y2(n), y3(n)])-20;
    min_z = min([z1(n), z2(n), z3(n)])-20;
    max_x = max([x1(n), x2(n), x3(n)])+20;
    max_y = max([y1(n), y2(n), y3(n)])+20;
    max_z = max([z1(n), z2(n), z3(n)])+20;
    axis([min_x max_x min_y max_y min_z max_z])

    %update simulation data
    body1.XData = x1(n);
    body1.YData = y1(n);
    body1.ZData = z1(n);
    body2.XData = x2(n);
    body2.YData = y2(n);
    body2.ZData = z2(n);
    body3.XData = x3(n);
    body3.YData = y3(n);
    body3.ZData = z3(n);
    addpoints(h1,x1(n),y1(n), z1(n));
    addpoints(h2,x2(n),y2(n), z2(n));
    addpoints(h3,x3(n),y3(n), z3(n));
   
    drawnow
    
    %end simulation if hit-time is reached
    if (n >= index_min_abstand)
        break;
    end
end


function err = to_minimize(v)
    %objective function which is given to the minimizer
    % returns the minimun distance between the satellit and dimorphos
    global dart_ode;
    global t_span;
    global y0;
    y_tmp = y0;
    y_tmp(2) = y_tmp(2)*v(1);
    y_tmp(4) = y_tmp(4)*v(1);
    y_tmp(6) = y_tmp(6)*v(1);
    
    options = odeset('RelTol',1e-5,'AbsTol',1e-7);
    [t,y] = ode45(dart_ode, t_span, y_tmp, options);
    
    distances = calc_distances(y);
    err = min(distances(:,1));
end

function distances = calc_distances(y)
    %calculates all the distances between the satellit and the 
    % two asteroids

    x1 = y(:, 1);
    y1 = y(:, 3);
    z1 = y(:, 5);
    
    x2 = y(:, 7);
    y2 = y(:, 9);
    z2 = y(:, 11);
    
    x3 = y(:, 13);
    y3 = y(:, 15);
    z3 = y(:, 17);
    distance_goal = sqrt((x3-x1).^2+(y3-y1).^2+(z3-z1).^2);
    distance_fail = sqrt((x2-x1).^2+(y2-y1).^2+(z2-z1).^2);
    distances = [distance_goal, distance_fail];
end


