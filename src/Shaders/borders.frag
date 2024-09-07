#version 330 core

#ifdef GL_ES
precision mediump float;
#endif

uniform vec2 cell_size;

void main() {
    vec2 st = gl_FragCoord.xy / cell_size;

//    float x = clamp(step(0.08, st.x) * (0.5 - step(0.9, st.x)) * 10.,0.0, 1.0);
//    float y = clamp(step(0.08, st.y) * (0.5 - step(0.9, st.y)) * 10.,0.0, 1.0);
//    vec3 color = 1. -vec3(x*y);
//
//    gl_FragColor = vec4(color, 1.0);

    // use floor division to index matrix? This might be incredibly bad
    gl_FragColor = vec4(fract(st.xy/cell_size.xy),0.0,1.0);

}