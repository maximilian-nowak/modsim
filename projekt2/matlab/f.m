
y0 = [0; 0; 0; 0; 0; 1; 1; 1; 0; 0; 1; 1; 1; 1; 0; 1; 0; 0];
[t,y] = ode45(@f2, [0 10],y0);

x1 = y(:, 1);
y1 = y(:, 3);
z1 = y(:, 5);

x2 = y(:, 7);
y2 = y(:, 9);
z2 = y(:, 11);

x3 = y(:, 13);
y3 = y(:, 15);
z3 = y(:, 17);
plot3(x1, y1, z1, 'b', 'LineWidth', 2);
hold on
plot3(x2, y2, z2, 'r', 'LineWidth', 2);
plot3(x3, y3, z3, 'g', 'LineWidth', 2);
grid on
xlabel('X');
ylabel('Y');
zlabel('Z');


function [dydt] = f2(t,y)
    G = 1; % vereinfachte Gravitationskonstante
    m = [1,1,1]; % vereinfachte Massen
    r12_3 = sqrt( (y( 1)-y( 7))^2 + (y( 3)-y( 9))^2 + (y( 5)-y(11))^2 )^3;
    r23_3 = sqrt( (y(13)-y( 7))^2 + (y(15)-y( 9))^2 + (y(17)-y(11))^2 )^3;
    r13_3 = sqrt( (y( 1)-y(13))^2 + (y( 3)-y(15))^2 + (y( 5)-y(17))^2 )^3;
    dydt = zeros(18,1);
    dydt( 1) = y( 2); dydt( 2) = G * ( m(2)*(y( 7)-y( 1))/r12_3 + m(3)*(y(13)-y( 1))/r13_3 );
    dydt( 3) = y( 4); dydt( 4) = G * ( m(2)*(y( 9)-y( 3))/r12_3 + m(3)*(y(15)-y( 3))/r13_3 );
    dydt( 5) = y( 6); dydt( 6) = G * ( m(2)*(y(11)-y( 5))/r12_3 + m(3)*(y(17)-y( 5))/r13_3 );
    dydt( 7) = y( 8); dydt( 8) = G * ( m(1)*(y( 1)-y( 7))/r12_3 + m(3)*(y(13)-y( 7))/r23_3 );
    dydt( 9) = y(10); dydt(10) = G * ( m(1)*(y( 3)-y( 9))/r12_3 + m(3)*(y(15)-y( 9))/r23_3 );
    dydt(11) = y(12); dydt(12) = G * ( m(1)*(y( 5)-y(11))/r12_3 + m(3)*(y(17)-y(11))/r23_3 );
    dydt(13) = y(14); dydt(14) = G * ( m(1)*(y( 1)-y(13))/r13_3 + m(2)*(y( 7)-y(13))/r23_3 );
    dydt(15) = y(16); dydt(16) = G * ( m(1)*(y( 3)-y(15))/r13_3 + m(2)*(y( 9)-y(15))/r23_3 );
    dydt(17) = y(18); dydt(18) = G * ( m(1)*(y( 5)-y(17))/r13_3 + m(2)*(y(11)-y(17))/r23_3 );
end