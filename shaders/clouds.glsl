#version 330

#define CLOUDS_ON
#define FADE

uniform vec2 iResolution;
uniform float iGlobalTime;

const int STEPS = 120;
const float STEPSIZE = 0.05;
const float DRAWDIST = STEPS * STEPSIZE;

const float PI = 3.1415926535897932384626433832795;
const float TWOPI = 2 * PI;

const int OCTAVES = 3;

struct ray
{
	vec3 o; //origin
	vec3 d;	//direction
};

vec3 calcCameraRayDir(float fov, vec2 fragCoord, vec2 resolution) 
{
	float fx = tan(radians(fov) / 2.0) / resolution.x;
	vec2 d = fx * (fragCoord * 2.0 - resolution);
	vec3 rayDir = normalize(vec3(d, 1.0));
	return rayDir;
}

float hash(vec3 p)
{
    p  = fract( p*0.3183099 + .1 );
	p *= 17.0;
    return fract( p.x*p.y*p.z*(p.x+p.y+p.z) );
}

float rand(float seed)
{
	return fract(sin(seed) * 1231534.9);
}

float rand(vec2 seed) 
{ 
    return rand(dot(seed, vec2(12.9898, 783.233)));
}

float noise( in vec3 x )
{
	x *= 2;
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
	
    return mix(mix(mix( hash(p+vec3(0,0,0)), 
                        hash(p+vec3(1,0,0)),f.x),
                   mix( hash(p+vec3(0,1,0)), 
                        hash(p+vec3(1,1,0)),f.x),f.y),
               mix(mix( hash(p+vec3(0,0,1)), 
                        hash(p+vec3(1,0,1)),f.x),
                   mix( hash(p+vec3(0,1,1)), 
                        hash(p+vec3(1,1,1)),f.x),f.y),f.z);
}

float fbm(vec3 p)
{
	p *= 1.4;
	float f = 0.0;
	float weight = 0.5;
	for(int i = 0; i < OCTAVES; ++i)
	{
		f += weight * noise( p );
		p.z -= iGlobalTime * i * 0.5;
		weight *= 0.5;
		p *= 2.0;
	}
	return f;
}

float density(vec3 p)
{
	p.y += 1.0;
	p.y += cos(p.x*1.4) * 0.2;
	p.y += cos(p.z)*0.1;
	p *= 1.2;
	p.z += iGlobalTime * 0.4;
	float noise = fbm(p);
	float clouds = noise*1.5 - p.y - 1.3;
	return clamp(clouds, 0, 1);
}

vec3 clouds(vec3 p, float d, float l, vec3 bg)
{
	vec3 lPos = vec3(0,0, DRAWDIST*1.7);
	vec3 lDir = lPos - p;

	float dL = density(p + normalize(lDir) * 0.2);
	float dG = clamp(d - dL, 0.0, 1.0);
	dG *= 1.0 - smoothstep(2,8, length(lDir));
	dG *= 70.0;
	vec3 cL = vec3(0, 0.1, 0.1) + vec3(1.0) * dG;
	vec3 cA = mix( vec3(1.0, 1.0, 1.0), vec3(1.0)*0.01, d);
	
	float a = 0.2;
	float t = exp(-a * l);	
	return mix(bg, cL * cA, t);
}

float stars(vec2 uv, float amount, float radius)
{
	uv = uv * amount;
	vec2 gridID = floor(uv);
	vec2 starPos = vec2(rand(gridID),rand(gridID+1));
	starPos = (starPos - 0.5) * 2.0;
	starPos = vec2(0.5) + starPos * (0.5 - radius * 2.0);
	float stars = distance(fract(uv), starPos);
	float size = rand(gridID)*radius;
	stars = 1.0 - smoothstep(size, size + radius, stars);
	return stars;
}

float gradient(vec2 uv)
{
	uv.x *= 0.8;
	uv *= 1.0 + sin(iGlobalTime*10) * 0.01;
	float g = clamp(1.0 - length(uv), 0, 1);
	return clamp(g, 0.0, 1.0);
}

float circle(vec2 uv, float r)
{
	return length(uv)-r;
}

float smin(float a, float b, float k)
{
	float h = clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0 );
    return mix( b, a, h ) - k*h*(1.0-h);
}

vec2 rotate(vec2 p, float angle)
{
	mat2 mat = mat2(cos(angle),-sin(angle),
					sin(angle),cos(angle));
	return p * mat;
}

float t(float scale, float k)
{
	float t = sin(iGlobalTime * scale);
	t = (t + 1.0)/2.0;
	t = mix(t,smoothstep(0, 1, t),k);
	t = (t - 0.5)*2.0;
	return t;
}

float ghost1(vec2 uv)
{
	float time = iGlobalTime * 6.0;
	float t = t(6.0, 0.5);

	uv.x += 0.5;
	uv = rotate(uv, t*max(0.0, uv.y)*0.2);
	uv.y -= 0.4 + sin(time * 2.0) * 0.1 * smoothstep(-0.5, 1.5, uv.y);
	vec2 originalUV = uv;
		
	uv.x *= 1.0 + uv.y;
	uv.y += max(0.0, -uv.y*0.8); 	
	float body = circle(uv, 0.2); 
	
	uv = originalUV;
	uv += vec2(-0.2, 0.2);
	uv = rotate(uv, -PI/4 + t*0.8*uv.x);
	uv *= vec2(0.4, 2.0);
	float arms = circle(uv, 0.1);
	
	uv = originalUV;
	uv += vec2(0.2, 0.2);
	uv = rotate(uv, PI/4 + t*0.8*(-uv.x));
	uv *= vec2(0.4, 2.0);
	arms = min(arms, circle(uv, 0.1));
	
	uv = originalUV;
	uv.x -= 0.01;
	uv.y += 0.05;
	uv.y *= 1.0 + cos(time*2)*0.4;
	float mouth = circle(uv, 0.02);
	
	uv = originalUV;
	uv.x -= 0.11;
	float eyeR = circle(uv, 0.02);
	uv.x += 0.2;
	float eyeL = circle(uv, 0.04);
	
	float d = body;
	d = smin(arms,body, 0.1);
	d = max(d, -eyeR);
	d = max(d, -eyeL);
	d = max(d, -mouth);
	float threshold = mix(0.04, 0.06, (0.5 +sin(iGlobalTime)*0.5));
	d = 1.0 - smoothstep(-threshold, threshold, d);	
	return d;
}

float ghost2(vec2 uv)
{
	uv.x -= 0.4;	
	uv.y += t(6.0, 0.5)*0.2* smoothstep(-1.0, 0.0, uv.y);
	vec2 originalUV = uv; 
	
	uv.x *= 1.0 + uv.y*0.4;
	uv.y *= mix(0.0, 1.0, smoothstep(-0.1, 0.0, uv.y));
	float body = circle(uv, 0.15);
	
	uv = originalUV;
	uv.x -= 0.06;
	float eyeR = circle(uv, 0.03);
	uv.x += 0.14;
	float eyeL = circle(uv, 0.025);
	
	float d = max(body,-eyeR);
	d = max(d, -eyeL);
	
	float threshold = mix(0.04, 0.06, (0.5 +sin(iGlobalTime)*0.5));
	d = 1.0 - smoothstep(-threshold, threshold, d);
	d *= 0.6;
	return d;
}

float ghosts(vec2 uv)
{	
	float d = ghost1(uv) + ghost2(uv);
	return clamp(d, 0.0, 1.0);
}

vec3 tonemapping(vec3 color, float exposure, float gamma)
{
	color *= exposure/(1. + color / exposure);
	color = pow(color, vec3(1. / gamma));
	float lum = 0.3*color.r + 0.6*color.g + 0.1*color.b;
	color = mix(color, color*color, 1.0 - smoothstep(0.0,0.4,lum));
	return color;
}

const vec2 FADE_T = vec2(0.0,4.0);
const vec2 FADE_D = vec2(1.0, 1.0);

float fadeIn(){return smoothstep(FADE_T.x, FADE_T.x+FADE_D.x,iGlobalTime);}
float fadeOut(){return smoothstep(FADE_T.y-FADE_D.y, FADE_T.y,iGlobalTime);}

vec3 fade(vec3 color)
{
	vec3 from = vec3(0.0);
	vec3 to = vec3(0.0);
	color = mix(from, color, fadeIn());
	color = mix(color, to, fadeOut());
	return color;
}

void main()
{
	vec2 coord = gl_FragCoord.xy;
	#ifdef FADE
		coord.y -= iResolution.y * (1.0 - fadeIn());
		coord.y += iResolution.y * fadeOut()*1.2;
	#endif
	
	vec2 res = vec2(max(iResolution.x, iResolution.y));
	vec2 uv = coord / res;
	uv = (uv-vec2(0.5))*2.0;
    uv.y += 0.5;
    uv *= 1.2;
			
	ray r;
	r.o = vec3(0.0);
	r.d = calcCameraRayDir(60, coord, res);
		
	float gradient = gradient(uv);
	float moon = distance(uv, vec2(0.0,0.1));
	moon = 1.0 - smoothstep(0.05, 0.08, moon);
	
	vec3 bg = mix(vec3(0.0, 0.1, 0.1),vec3(0.1, 0.3, 0.5), min(1.0, gradient*2.0));
	bg = mix(bg, vec3(0.6, 0.9, 1.0), (max(0.0, gradient - 0.5)) * 2.0);
	bg += vec3(0.8) * moon;
	bg += vec3(0.2) * stars(uv,5.0,0.01);
	bg += vec3(0.3) * stars(uv, 100.0, 0.04);
	bg += vec3(0.6) * ghosts(uv) * (uv.y+1.0)/2.0;
	
	vec4 sum = vec4(0);	
	float t = 0;
	#ifdef CLOUDS_ON
	for(int i = 0; i < STEPS; i++)
	{
		vec3 p = r.o + r.d * t;
		float d = density(p);
		if(d > 0.01)
		{
			float a = d * (1.0 - smoothstep(DRAWDIST / 2.0, DRAWDIST, t))*0.4;
			vec3 c = clouds(p, d, t, bg);
			sum += vec4(c * a, a) * ( 1.0 - sum.a );		
			if(sum.a > 0.99) break;
		}
		t += STEPSIZE;
	}	
	sum = clamp(sum,0.0,1.0);
	#endif
	vec4 c;
	c = vec4(bg, 1.0) * (1.0 - sum.a) + sum;
	c.rgb = tonemapping(c.rgb, 1.5,1.2);
	
	#ifdef FADE
	c.rgb = fade(c.rgb);
	#endif
	gl_FragColor = c;
}