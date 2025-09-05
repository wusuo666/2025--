% 脚本：可视化圆柱体是否被球体遮挡
% 此脚本提供一个使用示例，展示如何调用isCylinderBlockedBySphere函数
% 并可视化观察者、球体和圆柱体的三维场景

% 设置观察者和物体参数
observerPos = [11, 0, 0];                % 观察者位置
cylinderCenter = [0, 0, 0];             % 圆柱体底面中心
cylinderRadius = 1;                     % 圆柱体半径
cylinderHeight = 3;                     % 圆柱体高度
cylinderDirection = [0, 0, 1];          % 圆柱体方向（沿z轴）
sphereCenter = [6, 0, 0];               % 球体中心
sphereRadius = 1.5;                     % 球体半径

tic();
% 检查圆柱体是否被球体完全遮挡
result = isCylinderBlockedBySphere(observerPos, cylinderCenter, cylinderRadius, cylinderHeight, cylinderDirection, sphereCenter, sphereRadius);
toc();

% 输出结果
if result
    disp('圆柱体完全被球体遮挡');
else
    disp('圆柱体至少部分可见');
end

% 创建可视化场景
figure;
hold on;

% 绘制球体
[X, Y, Z] = sphere(30);
surf(X * sphereRadius + sphereCenter(1), Y * sphereRadius + sphereCenter(2), Z * sphereRadius + sphereCenter(3), ...
    'FaceAlpha', 0.3, 'EdgeColor', 'none', 'FaceColor', 'b');

% 绘制圆柱体
[X, Y, Z] = cylinder(cylinderRadius, 30);
Z = Z * cylinderHeight;
surf(X + cylinderCenter(1), Y + cylinderCenter(2), Z + cylinderCenter(3), ...
    'FaceAlpha', 0.3, 'EdgeColor', 'none', 'FaceColor', 'r');

% 绘制观察者
plot3(observerPos(1), observerPos(2), observerPos(3), 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 10);

% 绘制连接观察者和圆柱体的线
plot3([observerPos(1), cylinderCenter(1)], [observerPos(2), cylinderCenter(2)], [observerPos(3), cylinderCenter(3)], 'k--');

% 设置图形属性
axis equal;
grid on;
xlabel('X');
ylabel('Y');
zlabel('Z');
title('观察者、球体和圆柱体的三维视图');
view(30, 30);
legend('球体', '圆柱体', '观察者', '观察线');

% 允许用户交互修改场景参数
disp('可以修改脚本中的参数来尝试不同的场景配置：');
disp('- observerPos: 观察者位置');
disp('- cylinderCenter: 圆柱体底面中心');
disp('- cylinderRadius: 圆柱体半径');
disp('- cylinderHeight: 圆柱体高度');
disp('- cylinderDirection: 圆柱体方向');
disp('- sphereCenter: 球体中心');
disp('- sphereRadius: 球体半径'); 