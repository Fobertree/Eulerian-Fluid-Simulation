#version 330 core

#ifdef GL_ES
precision mediump float;
#endif

uniform vec2 cell_size;

//TODO
void main() {
    vec2 st = gl_FragCoord.xy / cell_size;

    gl_FragColor = vec4(st,0.0,1.0);

}