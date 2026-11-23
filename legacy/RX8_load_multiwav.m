function RX8_load_multiwav(wfile)

%cuts wav-file into chunks dependend on buffersize and creates tmp-files.


%%%% load and prep files %%%%
wavsize = wavread(wfile,'size');
buffsize = 360000;
samplrt = 12000;
semibuff = buffsize/2;
len = wavsize(1);
nchan = min(wavsize(2),8)
[sig,Fs] = wavread(wfile,1); %find out Samplerate(Fs)
chunksize = Fs/samplrt*semibuff;
nchunks = 0; rest = len;
while rest > 0
    nchunks = nchunks + 1;
    [sig,Fs] = wavread(wfile,[(nchunks-1)*chunksize+1 min(nchunks*chunksize,len)]);
    sig = resample(sig,samplrt,Fs)';
%???---------------
    sig = sig*8; %lautstaerke?
    if length(sig) < semibuff      
        sig = [sig, zeros(nchan,semibuff-length(sig))];
    end
    fid = fopen(sprintf('tmp4~%i.f32',nchunks),'w+');
    fwrite(fid,sig,'float32');
    fclose(fid);
    rest = rest - chunksize;
end

fprintf(1,'\nDone.\n\n');
