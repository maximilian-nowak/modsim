% Implementation of task 3
% This script simulates a six body system as an extension of the original
% three body problem.
% ---------------------------------------------------

close all;
clear;

format long;

% define initial values
% ---------------------
% format:
%   x; vx
%   y; vy
%   z; vz
    
y1 = [
% sun 1
    3; 0;
    3; 0;
    2; -0.02;
% planet1.1
    3; -1.5;
    3.4; 0;
    2; 0;
% planet1.2
    3; 1;
    2.2; 0;
    2; 0;
];

y2 = [
% sun 2
    6; 0;
    1; 0;
    1; 0.02;
% planet 2.1
    6; 1.5;
    0.6; 0;
    1; 0;
% planet 2.2
    6; -1;
    1.6; 0;
    1; 0
];
y2 = rotate_system(y2, -(5/12)*pi);    % rotate 2nd solar system by 75 degree

% set initial conditions
y0 = [y1;y2];                           % initial condition of ode
m = [1, 0.01, 0.01, 1, 0.01, 0.01];     % masses of bodys
f6b_ode = @(t, y)N_body_ode(t,y,m);     % anonymous func to pass on the masses to the ode

% 1. run:  determine time of collision
Opt    = odeset('Events', @solar_collision);
[tcol, ycol] = ode45(f6b_ode, 0:0.01:25, y0, Opt);
t_span = 0:0.01:tcol(end)*2;

% 2. run: full simulation with doubled time span
[t,y] = ode45(f6b_ode, t_span ,y0);


% Plot the results
% ----------------

% plot initial graphic objects
body1 = plot3(y(1, 1), y(1, 3), y(1, 5), 'r.', 'MarkerSize', 30);
hold on
body2 = plot3(y(1, 7), y(1, 9), y(1, 11), 'm.', 'MarkerSize', 13);
body3 = plot3(y(1, 13), y(1, 15), y(1, 17), 'm.', 'MarkerSize', 13);
body4 = plot3(y(1, 19), y(1, 21), y(1, 23), 'b.', 'MarkerSize', 30);
body5 = plot3(y(1, 25), y(1, 27), y(1, 29), 'c.', 'MarkerSize', 13);
body6 = plot3(y(1, 31), y(1, 33), y(1, 35), 'c.', 'MarkerSize', 13);
h1 = animatedline('Color','r', 'MaximumNumPoints', 300);
h2 = animatedline('Color','m', 'MaximumNumPoints', 200, 'LineStyle', '--');
h3 = animatedline('Color','m', 'MaximumNumPoints', 200, 'LineStyle', ':');
h4 = animatedline('Color','b', 'MaximumNumPoints', 300);
h5 = animatedline('Color','c', 'MaximumNumPoints', 200, 'LineStyle', '--');
h6 = animatedline('Color','c', 'MaximumNumPoints', 200, 'LineStyle', ':');

grid on
xlabel('X');
ylabel('Y');
zlabel('Z');
description=['computed collision at t=', num2str(round(tcol(end), 0))];
title({description});

axis([0 10 0 10 0 5]);
view_xy = 0;
view_rising = false;
view([view_xy view_xy])

% create video writer and start recording
videoFile = 'animation.mp4';
fps = 30; % fps
video = VideoWriter(videoFile, 'MPEG-4');
video.FrameRate = fps;
open(video);

% start animation of the trajectories
for n=2:length(y(:, 1))
    % update view angle 
    if (view_rising==false) && (view_xy >= -45)
        view_xy = view_xy-0.5;
    elseif view_rising && (view_xy <= 45)
        view_xy = view_xy+0.5;
    else
        view_rising = ~view_rising;
    end
    view([view_xy view_xy]);
   
    % center everything around the mean of the two suns
    y(n, 1:6:36) = y(n, 1:6:36) - mean(y(n, 1:18:36));  % x coords
    y(n, 3:6:36) = y(n, 3:6:36) - mean(y(n, 3:18:36));  % y coords
    y(n, 5:6:36) = y(n, 5:6:36) - mean(y(n, 5:18:36));  % z coords
    
    % adjust axis scales according to min and max values
    axis([ ...
        min(y(n, 1:6:36))-0.5 max(y(n, 1:6:36))+0.5 ...
        min(y(n, 3:6:36))-0.5 max(y(n, 3:6:36))+0.5 ...
        min(y(n, 5:6:36))-0.5 max(y(n, 5:6:36))+0.5 ...
    ]);
    
    % update plots objects
    body1.XData = y(n, 1);
    body1.YData = y(n, 3);
    body1.ZData = y(n, 5);
    body2.XData = y(n, 7);
    body2.YData = y(n, 9);
    body2.ZData = y(n, 11);
    body3.XData = y(n, 13);
    body3.YData = y(n, 15);
    body3.ZData = y(n, 17);
    body4.XData = y(n, 19);
    body4.YData = y(n, 21);
    body4.ZData = y(n, 23);
    body5.XData = y(n, 25);
    body5.YData = y(n, 27);
    body5.ZData = y(n, 29);
    body6.XData = y(n, 31);
    body6.YData = y(n, 33);
    body6.ZData = y(n, 35);
    addpoints(h1,y(n, 1),y(n, 3), y(n, 5));
    addpoints(h2,y(n, 7),y(n, 9), y(n, 11));
    addpoints(h3,y(n, 13),y(n, 15), y(n, 17));
    addpoints(h4,y(n, 19),y(n, 21), y(n, 23));
    addpoints(h5,y(n, 25),y(n, 27), y(n, 29));
    addpoints(h6,y(n, 31),y(n, 33), y(n, 35));

    description=['computed collision at t=', num2str((-1)*tcol(end) + t(n), '%.2f')];
    title({description});
    drawnow;
    
    % captures current frame in video
    frame = getframe(gcf);
    writeVideo(video, frame);
end

close(video);  % stop recording


% utility functions for the script
% ---------------------------------

function y = rotate_system(y, angle)
%rotate_system Rotates planets in the given solar system around y axis.
% args:
%   y       - solar system of two planets and a sun.
%   angle   - rotation angle in radians in terms of pi.

    R75 = [cos(angle) 0 sin(angle); ...  
          0 1 0; ...
          -sin(angle) 0 cos(angle)];

    % apply rotation to planet 1
    x1 = (R75 * (y(7:2:12) - y(1:2:6))) + y(1:2:6);
    v1 = R75 * y(8:2:12);
    y1 = reshape([x1 v1].', 1, []);

    % apply rotation to planet 2
    x2 = (R75 * (y(13:2:18) - y(1:2:6))) + y(1:2:6);
    v2 = R75 * y(14:2:18);
    y2 = reshape([x2 v2].', 1, []);

    % merge new values back together
    y = [y(1:6); y1'; y2'];
end

function [value, isterminal, direction] = solar_collision(~, y)
%solar_collision event which terminates computation of ode45 when 
%   collision of suns is detected.

    idx_sun1 = 1:2:6;
    idx_sun2 = 19:2:23;
    tol = 1e-2;
    
    % computation will stop when value is set to 0.
    if (norm(y(idx_sun1) - y(idx_sun2)) < tol)
        value = 0;
    else
        value = 1;
    end
    
    isterminal = 1;
    direction  = 0;
end
