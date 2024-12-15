clear all;
close all;

y0 = [
    2; 0;
    2; 0; 
    1; 0;
    
    0.5; 0.5;
    0.5; 0;
    1; 0;
    
    1.5; 0.5;
    1; 0;
    1; 0;

    10; 0;
    2; 0; 
    1; 0;

    8.5; 0.5;
    0.5; 0;
    1; 0.2;

    9.5; 0.5;
    1; 0;
    1; 0.2
];

m = [1, 0.01, 0.01, 1, 0.01, 0.01];
f6b_ode = @(t, y)N_body_ode(t,y,m);

[t,y] = ode45(f6b_ode, [0:0.05:25],y0);

x1 = y(:, 1);
y1 = y(:, 3);
z1 = y(:, 5);

x2 = y(:, 7);
y2 = y(:, 9);
z2 = y(:, 11);

x3 = y(:, 13);
y3 = y(:, 15);
z3 = y(:, 17);

x4 = y(:, 19);
y4 = y(:, 21);
z4 = y(:, 23);

x5 = y(:, 25);
y5 = y(:, 27);
z5 = y(:, 29);

x6 = y(:, 31);
y6 = y(:, 33);
z6 = y(:, 35);


body1 = plot3(x1, y1, z1, 'b.', 'MarkerSize', 30)
hold on
body2 = plot3(x2, y2, z2, 'r.', 'MarkerSize', 10)
body3 = plot3(x3, y3, z3, 'r.', 'MarkerSize', 10)
body4 = plot3(x4, y4, z4, 'b.', 'MarkerSize', 30)
body5 = plot3(x5, y5, z5, 'g.', 'MarkerSize', 10)
body6 = plot3(x6, y6, z6, 'g.', 'MarkerSize', 10)
h1 = animatedline('Color','b', 'MaximumNumPoints', 200);
h2 = animatedline('Color','r', 'MaximumNumPoints', 200, 'LineStyle', '--');
h3 = animatedline('Color','r', 'MaximumNumPoints', 200, 'LineStyle', ':');
h4 = animatedline('Color','b', 'MaximumNumPoints', 200);
h5 = animatedline('Color','g', 'MaximumNumPoints', 200, 'LineStyle', '--');
h6 = animatedline('Color','g', 'MaximumNumPoints', 200, 'LineStyle', ':');

grid on
xlabel('X');
ylabel('Y');
zlabel('Z');
axis([0 10 0 10 0 5]);
view_xy = 0;
view_rising = false;

for n=1:length(y(:, 1))
    if (view_rising==false) && (view_xy >= -45)
        view_xy = view_xy-0.5;
    elseif view_rising && (view_xy <= 45)
        view_xy = view_xy+0.5;
    else
        view_rising = ~view_rising;
    end
    view([view_xy view_xy])
    min_x = min([y(n, 13), y(n, 7), y(n, 1),y(n, 19), y(n, 25), y(n, 31)])-3;
    min_y = min([y(n, 3), y(n, 9), y(n, 15), y(n, 21), y(n, 27), y(n, 33)])-3;
    min_z = min([y(n, 11), y(n, 5), y(n, 17), y(n, 23), y(n, 29), y(n, 35)])-3;
    max_x = max([y(n, 13), y(n, 7), y(n, 1),y(n, 19), y(n, 25), y(n, 31)])+3;
    max_y = max([y(n, 3), y(n, 9), y(n, 15), y(n, 21), y(n, 27), y(n, 33)])+3;
    max_z = max([y(n, 11), y(n, 5), y(n, 17), y(n, 23), y(n, 29), y(n, 35)])+3;
    axis([min_x max_x min_y max_y min_z max_z])
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

    drawnow
%     pause(0.01);
end
