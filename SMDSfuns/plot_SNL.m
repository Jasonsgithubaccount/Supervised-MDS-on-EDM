function  plot_SNL(A,B,a,b,pars)
% Graph embedding ponits
figure,
if ~isfield(pars,'E')   
    PP=pars.PP;
    if isfield(pars, 'm'); m=pars.m; else m=0;  end
    [r,n]= size(PP);
    T   = 1:m;
    T1  = m+1:n;
    if r==2
        Af=[PP(:,T) A]; 
        Bf=[PP(:,T) B];
        for j=1:2
            if j==2; A=B; Af=Bf; a=b; end
            subplot(1,2,j),
            set(gca,'FontName','Times','FontSize',8)
            plot(PP(1,T),PP(2,T),'gs','markersize',4,'linewidth',2);     
            for i=1:n-m; 
                line([A(1,i) PP(1,m+i)], [A(2,i) PP(2,m+i)],...
                    'linewidth',.1,'color','b'); hold on
            end
            plot(PP(1,T1),PP(2,T1),'bo','markersize',4.5);hold on
            plot(A(1,:),A(2,:),'m*','markersize',3);hold on          
            plot(PP(1,T),PP(2,T),'gs','markersize',4,'linewidth',2);
            ZZ = [PP Af]';
            if j==1;
                xlabel(['Before refinement: RMSD = ', sprintf('%4.2e', a)],...
                    'FontName','Times','FontSize',8);
            else
                xlabel(['After refinement: rRMSD = ', sprintf('%4.2e', a)],...
                    'FontName','Times','FontSize',8);
            end
            axis([min(min(ZZ(:,1))) max(max(ZZ(:,1))) min(min(ZZ(:,2))) max(max(ZZ(:,2)))]) 
            hold on
        end
    else
        Af=[PP(:,T) A];
        Bf=[PP(:,T) B];  
        for j=1:2
            if j==2; A=B; Af=Bf; a=b; end
            subplot(1,2,j),       
            set(gca,'FontName','Times','FontSize',8)
           
            plot3(PP(1,T1),PP(2,T1),PP(3,T1),'bo','markersize',4.5);hold on
            plot3(A(1,:),A(2,:),A(3,:),'m*','markersize',3);hold on   
            plot3(PP(1,T),PP(2,T),PP(3,T),'gs','markersize',4,'linewidth',2); hold on            
            for i=1:n-m; 
                line([A(1,i) PP(1,m+i)], [A(2,i) PP(2,m+i)], [A(3,i) PP(3,m+i)],...
                    'linewidth',.1,'color','b'); hold on
            end
            ZZ = [PP Af]';
            plot3(PP(1,T),PP(2,T),PP(3,T),'gs','markersize',4,'linewidth',2); hold on        
            if j==1;
                title(['Before refinement: RMSD = ', sprintf('%4.2e', a)],...
                    'FontName','Times','FontSize',8);
            else
                title(['After refinement: rRMSD = ', sprintf('%4.2e', a)],...
                    'FontName','Times','FontSize',8);
            end
            grid on;
            axis([min(min(ZZ(:,1))) max(max(ZZ(:,1))) min(min(ZZ(:,2))) max(max(ZZ(:,2)))...
                min(min(ZZ(:,3))) max(max(ZZ(:,3)))]) 
            hold on             
        end
    end    
else
    if a <b; B=A; end
    Xfrefine=[pars.PP(:,1:pars.m) B];
    scatter(Xfrefine(1,pars.E),Xfrefine(2,pars.E),12,'filled');
	hold on
	scatter(Xfrefine(1,pars.D),Xfrefine(2,pars.D),12,'filled');
	scatter(Xfrefine(1,pars.M),Xfrefine(2,pars.M),12,'filled');
 	scatter(Xfrefine(1,1:pars.m),Xfrefine(2,1:pars.m),12,'filled','k');
    set(gca,'FontName','Times','FontSize',8);
    if a <b;          
        title(['Before refinement: RMSD = ',  sprintf('%4.2e', a)],...
                    'FontName','Times','FontSize',9);  
    else
        title(['After refinement: rRMSD = ', sprintf('%4.2e', b)],...
                    'FontName','Times','FontSize',9);
    end
	axis equal;
	axis([-5 120 -5 55]);
end
    
end