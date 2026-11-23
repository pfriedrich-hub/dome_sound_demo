function RX8_play_multichannel_develop(ntfile,RX8,zBus)
% help? ask Marc!
% RX8_play_multichannel_develop(nchan,nchunks,RX8,zBus)
% Load RX8 and zBus with [RX8,zBus] = RX8init_8Sig!
% If tmp files not there, run RX8_load_multiwav(wfile);
% Check Buffersize! save changed values!!RESTART RX8s!!!

buffsize = 360000;
samplrt = 12000;
semibuff = buffsize/2;

%%%% Coordinates/Speakerpositions %%%%

jwd = load('C:\Users\Dome\01_good files\textfiles\RX_cab_pos.txt');
pos = jwd(:,4:5); 
[posx(:,1),posx(:,2),posx(:,3)] = sph2cart(pos(:,1)*pi/180,pos(:,2)*pi/180,95); % xyz coordinates
layout = jwd(:,6:7); %Speakerpositions on RX8s

% loding first two chunks to buffers (done now to ger nchans)
fprintf(1,'\nsend package1+2\n\n')
fid = fopen(sprintf('tmp%i~1.f32',ntfile),'r');
sigA = fread(fid,'float32');
nchan = length(sigA)/semibuff;
sigA = reshape(sigA,nchan,semibuff);
fclose(fid);

fid = fopen(sprintf('tmp%i~2.f32',ntfile),'r');
sigB = fread(fid,'float32');
sigB = reshape(sigB,nchan,semibuff);
fclose(fid);

%%%% setup GUI %%%%            
fig = figure();                
set(fig,'Tag','RX_control');  
                               
for I=1:80
    ellipse(I) = annotation('ellipse','FaceColor',[0 0 0],'Position',[(pos(I,1)/360)+0.5,(pos(I,2)/360)+0.5,.02,.02]);
end

for I = 1:nchan
    figsig(I) = scribe.textbox('HorizontalAlignment','center','String',sprintf('S%i',I),'Position',[rand,rand,.05,.05]);
end
drawnow;

%%%% send wavs to RX8 and start %%%%
% get nchunks
nchunks = 1;
while exist(sprintf('tmp%i~%i.f32',ntfile,nchunks),'file');
    nchunks = nchunks + 1;
end
nchunks = nchunks - 1;

zBus.zBusTrigB(0,1,10); %Reset Buffer to prevent playing when invoke 'Run'

for I = 1:nchan   
    for RX = 1:4
        invoke(RX8(RX),'WriteTagV',sprintf('sigin%i',I),0,sigA(I,:)); %***
        invoke(RX8(RX),'WriteTagV',sprintf('sigin%i',I),semibuff,sigB(I,:)); %***
        invoke(RX8(RX),'Run');
    end
    fprintf(1,'channel %i loaded\n',I);
   
    
end
%time=etime(clock,t);
input('Press RETURN to start\n');

%fprintf(1,'wait 5 seconds\n');
%pause(5);
%t = clock;
%curindex = invoke(RX8(1),'GetTagVal','idx1')

zBus.zBusTrigB(0,2,10);
pause(0.5); %because auf possible zBus-Trigger latency (-->RPvdsEX-Help)
zBus.zBusTrigA(0,2,10); %start Buffer-reading
%curindex = invoke(RX8(1),'GetTagVal','idx1')
zBus.zBusTrigA(0,1,10);

%%%% loop %%%%
fwhm = 150; s = (fwhm/2.3548).^2;  % determine width of signal
amps = zeros(nchan,80);
ntmp = 3; %tmp-counter
runflag = 1;
stopflag = 0;
flag = 1;

while runflag
    % get all positions and widths
    for I = 1:nchan
        
        jwd = get(figsig(I),'Position');
        tboxpos = jwd(1:2); % tboxsize(I,3) = max(jwd(3:4));
        loc = (tboxpos-0.5)*360;
        [locx(1),locx(2),locx(3)] = sph2cart(loc(1)*pi/180,loc(2)*pi/180,95);
        % calculate distances and sigma
        d = sqrt(sum((posx-repmat(locx,80,1)).^2,2));
        amps(I,:) = (1/sqrt(2*pi*s))*exp(-(d.^2)/(2*s));
        amps(I,:) = amps(I,:)./sum(amps(I,:))*1; % normalize sum of contributions from each channel to 1
        % set amps
        for a = 1:80 
            RX = layout(a,1);
            chan = layout(a,2);
            invoke(RX8(RX),'SetTagVal',sprintf('aSig%i_Ch%i',I,chan),amps(I,a));
            set(ellipse(a),'FaceColor',[min(sum(amps(:,a))*10,1) min(sum(amps(:,a))*5,1) min(sum(amps(:,a)),1)]);
        end
        
        % check curr_idx
        curindex = invoke(RX8(1),'GetTagVal','idx1');
       
        %write to BufferA
            if (curindex>semibuff) && flag==1; %Wenn idx in BufferB & alte tmp noch in BufferA
                if stopflag == 0 
                    %time = etime(clock, t);
                    fid = fopen(sprintf('tmp%i~%i.f32',ntfile,ntmp),'r'); %ntmp an richtiger Stelle?
                    sigA = fread(fid,'float32');
                    sigA = reshape(sigA,nchan,semibuff);
                    fclose(fid);
                    %%% loop thru RX8s and spuren
                    fprintf(1,'loading chunk %i into sigA.\n',ntmp);
                    for RX = 1:4
                        for spur = 1:nchan
                            %invoke(RX8(RX),'WriteTagV',sprintf('sigin%i',I),0,sigA(spur,:));
                            invoke(RX8(RX),'WriteTagV',sprintf('sigin%i',spur),0,sigA(spur,:));
                        end
                    end
                    ntmp = ntmp+1;
                    flag = 0;
                else
                    runflag = 0;
                end
            end 
    
        % write to BufferB 
            if (curindex<semibuff) && flag==0; %Wenn idx in BufferA & alte tmp noch in BufferB
                if stopflag == 0 
                    %time = etime(clock, t);
                    fid = fopen(sprintf('tmp%i~%i.f32',ntfile,ntmp),'r');
                    sigB = fread(fid,'float32');
                    sigB = reshape(sigB,nchan,semibuff);
                    fclose(fid);
                    %%% loop thru RX8s and spuren
                    fprintf(1,'loading chunk %i into sigB.\n',ntmp);
                    for RX = 1:4
                        for spur = 1:nchan
                            %invoke(RX8(RX),'WriteTagV',sprintf('sigin%i',I),semibuff,sigB(spur,:));
                            invoke(RX8(RX),'WriteTagV',sprintf('sigin%i',spur),semibuff,sigB(spur,:));
                        end
                    end
                    ntmp = ntmp+1;
                    flag = 1;
                else
                    runflag = 0;
                end
            end 
            
    if ntmp > nchunks
        stopflag = 1;
    end
    end
    pause(0.001);
end

pause(semibuff/samplrt); %wait until last chunk played

%%%% stop %%%%
zBus.zBusTrigA(0,2,10); % stop playing
for RX = 1:4
   
   invoke(RX8(RX),'Halt');
end
% delete tmp files
%
%for i = 1:nchunks
%fid = fopen(sprintf('tmp~%i.f32',i),'w+');
%    fclose(fid);
%    system(sprintf('del tmp~%i.f32',i));
%end 
fprintf(1,'\nDone.\n\n');
