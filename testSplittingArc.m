% Try splitting algorithm for morphogenesis project, based on making single
% linear slice through domain, parametrised by postion along theta-axis and
% angle to normal
% Kyle Wedgwood
% 30.1.2019

function testSplittingArc( no_real)

  if nargin < 1
    no_real = 1;
  end

  % Generate mesh
  N = 10000;
  rho_max = 1.0;
  psi_max = 2.0/3.0*pi;
  
  [psi,rho,x_pos,y_pos] = assignCellPositions( N, psi_max, rho_max);

  for real_no = 1:no_real
    % sameSharpness
    % nonlinearMonotonic
    % nonlinearNonMonotonic
    fate = assignCellFates( psi, rho, psi_max, 'nonlinearNonMonotonic');
    %plotFates( x_pos, y_pos, fate);
    pause(0.1);

    % Save coordinates
    coords = [ x_pos, y_pos, fate ];

    if ~exist( 'D:\\CytonemeSignaling\\testDataStudySharpness_linear\\nonlinearNonMonotonic\\', 'dir')
      mkdir( 'D:\\CytonemeSignaling\\testDataStudySharpness_linear\\nonlinearNonMonotonic\\');
    end
    
    files = dir( 'D:\\CytonemeSignaling\\testDataStudySharpness_linear\\nonlinearNonMonotonic\\');
    filenames = {};

    % Existing files in directory
    for j = 1:length( files)
      if ~files(j).isdir
        filenames = [ filenames; files(j).name];
      end
    end

    % Initialise
    i = 1;
    filename = sprintf( 'coords_%05d.dat', i);

    while ismember( filename, filenames)
      i = i+1;
      filename = sprintf( 'coords_%05d.dat', i);
    end
      
    save( sprintf( 'D:\\CytonemeSignaling\\testDataStudySharpness_linear\\nonlinearNonMonotonic\\%s', filename), 'coords', '-ascii');
  end
    
end

function [psi,rho,x_pos,y_pos] = assignCellPositions( N, psi_max, rho_max)

  rho = rho_max*rand( N, 1); % a list of rho to be used
  psi = psi_max*rand( N, 1); % a list of p to be used
  
  x_pos = ( 1.0+rho).*cos( -psi+pi/2.0);
  y_pos = ( 1.0+rho).*sin( psi+pi/2.0);
    
end

function fate = assignCellFates( psi, rho, psi_max, type)

  u = rand( size( psi));
  %%%%%
  % change sharpness here
  %%%%%
  expsharp = 0.05
  switch type
    
    case 'sameSharpness'
      
      while 1
        thresh = psi_max*rand( 2, 1);
        thresh = sort( thresh);
        if (diff( thresh) > 0.2) && (thresh(1) > 0.3) && (thresh(2) < psi_max-0.3)
          break;
        end
      end
      
      sharp = expsharp*ones( 2, 1);  % the bigger the less sharp£¬ tune this parametre in experiment
      % psi_max = 2.094
      % sharp scale the shape of tanh
      % smaller than -0.549 or not in tanh()
      % thresh(1) < thresh(2)
      
      % tanh()
      % psi - thresh
      % /sharp
      p1 = 0.5*( 1.0 - tanh( psi - thresh(1))/sharp(1));
      p2 = 0.5*( 1.0 - tanh( psi - thresh(2))/sharp(2));
      disp([p1 p2])
    case 'sameLocation'
      
      thresh = psi_max/3.0*[1;2];
      sharp  = 0.3*rand( 2, 1);

      p1 = 0.5*( 1.0 - tanh( (psi-thresh(1))/sharp(1)));
      p2 = 0.5*( 1.0 - tanh( (psi-thresh(2))/sharp(2)));
      
    case 'nonlinearMonotonic'
      
      thresh = psi_max/3.0*[1;2];
      sharp  = expsharp*ones( 2, 1);
      
      % tanh()
      % psi - 0.5*rho - thresh
      % /sharp
      p1 = 0.5*( 1.0 - tanh( (psi-0.5*rho-thresh(1))/sharp(1)));
      p2 = 0.5*( 1.0 - tanh( (psi-0.5*rho-thresh(2))/sharp(2)));
      disp([p1 p2])
    case 'nonlinearNonMonotonic'
      
      thresh = psi_max/3.0*[1;2];
      sharp  = expsharp*ones( 2, 1);
      
      % tanh()
      % psi - (0.5*rho)^2 - thresh
      % /sharp
      p1 = 0.5*( 1.0 - tanh( (psi-( rho-0.5).^2-thresh(1))/sharp(1)));
      p2 = 0.5*( 1.0 - tanh( (psi-( rho-0.5).^2-thresh(2))/sharp(2)));
      disp([p1 p2])
  end
 
  fate = (u<p2).*(u>p1) + 2*(u>p2);
  
end

function f = plotFates( x_pos, y_pos, fate)

    f = figure;

    plot( x_pos(fate==0), y_pos(fate==0), 'ro', ...
          x_pos(fate==1), y_pos(fate==1), 'bo', ...
          x_pos(fate==2), y_pos(fate==2), 'go');
    axis equal
    axis off;
    drawnow;
    
end