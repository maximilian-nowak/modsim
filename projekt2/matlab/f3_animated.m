% Implementation of task 1
% This script simulates the long-term effects of the three body problem
% ---------------------------------------------------

clear all;
close all;

%starting point
y0 = [
    1; 0;
    0; 0; 
    0; 0.1;
    
    0; 0.1;
    1; 0;
    0; 0;
    
    0; 0;
    0; 0.1;
    1; 0;
];
%second startpoint with small offset
y1 = y0;
y1(1) = y1(1)+10^-15;

%simple mass
m = [1, 1, 1];
f3b_ode = @(t, y)N_body_ode(t,y,m);

%simulate both systems with same time intervall
t_max = 10;
options = odeset('RelTol',1e-5,'AbsTol',1e-7);
[t_2,y2_] = ode45(f3b_ode, [0:0.001:t_max],y1, options);

[t,y] = ode45(f3b_ode,[0:0.001:t_max],y0, options);


%get all coordinates from ode return value for first simulation
x1 = y(:, 1);
y1 = y(:, 3);
z1 = y(:, 5);

x2 = y(:, 7);
y2 = y(:, 9);
z2 = y(:, 11);

x3 = y(:, 13);
y3 = y(:, 15);
z3 = y(:, 17);

%get all coordinates from ode return value for second simulation
x_1 = y2_(:, 1);
y_1 = y2_(:, 3);
z_1 = y2_(:, 5);

x_2 = y2_(:, 7);
y_2 = y2_(:, 9);
z_2 = y2_(:, 11);

x_3 = y2_(:, 13);
y_3 = y2_(:, 15);
z_3 = y2_(:, 17);





%configuration for animation
line_length = 200;
body1 = plot3(x1, y1, z1, 'r.', 'MarkerSize', 20);
hold on
body2 = plot3(x2, y2, z2, 'g.', 'MarkerSize', 20);
body3 = plot3(x3, y3, z3, 'b.', 'MarkerSize', 20);
body_1 = plot3(x_1, y_1, z_1, 'r.', 'MarkerSize', 20);
body_2 = plot3(x_2, y_2, z_2, 'g.', 'MarkerSize', 20);
body_3 = plot3(x_3, y_3, z_3, 'b.', 'MarkerSize', 20);
h1 = animatedline('Color','r', 'MaximumNumPoints', line_length, 'LineWidth',2);
h2 = animatedline('Color','g', 'MaximumNumPoints', line_length, 'LineWidth',2);
h3 = animatedline('Color','b', 'MaximumNumPoints', line_length, 'LineWidth',2);
h_1 = animatedline('Color','r', 'MaximumNumPoints', line_length, 'LineStyle','--', 'LineWidth',1);
h_2 = animatedline('Color','g', 'MaximumNumPoints', line_length, 'LineStyle','--', 'LineWidth',1);
h_3 = animatedline('Color','b', 'MaximumNumPoints', line_length, 'LineStyle','--', 'LineWidth',1);

grid on
xlabel('X');
ylabel('Y');
zlabel('Z');
axis_ofset = 1;

% start animation
for n=1:15:length(y(:, 1))
    title(sprintf('Time: %0.5f',t(n)));
    %calc new coordinate origin
    min_x = min([x1(n), x2(n), x3(n)])-axis_ofset;
    min_y = min([y1(n), y2(n), y3(n)])-axis_ofset;
    min_z = min([z1(n), z2(n), z3(n)])-axis_ofset;
    max_x = max([x1(n), x2(n), x3(n)])+axis_ofset;
    max_y = max([y1(n), y2(n), y3(n)])+axis_ofset;
    max_z = max([z1(n), z2(n), z3(n)])+axis_ofset;
    axis([min_x max_x min_y max_y min_z max_z])
    %update first simulation data
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
  
    %update second simulation data
    body_1.XData = x_1(n);
    body_1.YData = y_1(n);
    body_1.ZData = z_1(n);
    body_2.XData = x_2(n);
    body_2.YData = y_2(n);
    body_2.ZData = z_2(n);
    body_3.XData = x_3(n);
    body_3.YData = y_3(n);
    body_3.ZData = z_3(n);
    addpoints(h_1,x_1(n),y_1(n), z_1(n));
    addpoints(h_2,x_2(n),y_2(n), z_2(n));
    addpoints(h_3,x_3(n),y_3(n), z_3(n));
    
    drawnow
    %pause(0.02);
end
legend('Körper1_1','Körper2_1', 'Körper3_1', 'Körper1_2', 'Körper2_2', 'Körper3_2', 'Körper1_1','Körper2_1', 'Körper3_1', 'Körper1_2', 'Körper2_2', 'Körper3_2')



