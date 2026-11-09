/* Offline RGB24 stdin -> H.264 MP4 using already-installed FFmpeg libraries.
 * Build: cc encode_rgb_mp4.c -o encode_rgb_mp4 $(pkg-config --cflags --libs libavcodec libavformat libavutil libswscale)
 * Usage: encode_rgb_mp4 OUTPUT WIDTH HEIGHT FPS < RGB24_FRAMES
 */
#include <stdio.h>
#include <stdlib.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
static void check(int code) { if(code<0){char msg[256];av_strerror(code,msg,sizeof msg);fprintf(stderr,"%s\n",msg);exit(1);} }
static void drain(AVCodecContext *c, AVFormatContext *o, AVStream *s, AVPacket *p){
 int e;while((e=avcodec_receive_packet(c,p))>=0){av_packet_rescale_ts(p,c->time_base,s->time_base);p->stream_index=s->index;check(av_interleaved_write_frame(o,p));av_packet_unref(p);}if(e!=AVERROR(EAGAIN)&&e!=AVERROR_EOF)check(e);
}
int main(int argc,char **argv){
 if(argc!=5)return 2;int w=atoi(argv[2]),h=atoi(argv[3]),fps=atoi(argv[4]);if(w<=0||h<=0||fps<=0||w%2||h%2)return 2;
 AVFormatContext *o=NULL;check(avformat_alloc_output_context2(&o,NULL,"mp4",argv[1]));const AVCodec *codec=avcodec_find_encoder_by_name("libx264");if(!codec){fprintf(stderr,"Installed libavcodec lacks libx264\n");return 3;}
 AVStream *s=avformat_new_stream(o,NULL);if(!s)return 3;AVCodecContext *c=avcodec_alloc_context3(codec);if(!c)return 3;c->width=w;c->height=h;c->pix_fmt=AV_PIX_FMT_YUV420P;c->time_base=(AVRational){1,fps};c->framerate=(AVRational){fps,1};c->gop_size=fps*2;if(o->oformat->flags&AVFMT_GLOBALHEADER)c->flags|=AV_CODEC_FLAG_GLOBAL_HEADER;
 check(av_opt_set(c->priv_data,"preset","veryfast",0));check(av_opt_set(c->priv_data,"crf","23",0));check(avcodec_open2(c,codec,NULL));check(avcodec_parameters_from_context(s->codecpar,c));s->time_base=c->time_base;
 check(avio_open(&o->pb,argv[1],AVIO_FLAG_WRITE));AVDictionary *opts=NULL;av_dict_set(&opts,"movflags","+faststart",0);check(avformat_write_header(o,&opts));av_dict_free(&opts);
 AVFrame *f=av_frame_alloc();f->format=c->pix_fmt;f->width=w;f->height=h;check(av_frame_get_buffer(f,32));AVPacket *p=av_packet_alloc();struct SwsContext *sw=sws_getContext(w,h,AV_PIX_FMT_RGB24,w,h,AV_PIX_FMT_YUV420P,SWS_BILINEAR,NULL,NULL,NULL);if(!sw)return 3;
 size_t n=(size_t)w*h*3;unsigned char *rgb=malloc(n);int64_t i=0;size_t got;
 while((got=fread(rgb,1,n,stdin))==n){check(av_frame_make_writable(f));const unsigned char *src[1]={rgb};int stride[1]={w*3};sws_scale(sw,src,stride,0,h,f->data,f->linesize);f->pts=i++;check(avcodec_send_frame(c,f));drain(c,o,s,p);}
 if(got){fprintf(stderr,"Partial RGB frame\n");return 4;}check(avcodec_send_frame(c,NULL));drain(c,o,s,p);check(av_write_trailer(o));fprintf(stderr,"Encoded %ld frames at %d fps\n",(long)i,fps);
 free(rgb);sws_freeContext(sw);av_packet_free(&p);av_frame_free(&f);avcodec_free_context(&c);avio_closep(&o->pb);avformat_free_context(o);return 0;
}
