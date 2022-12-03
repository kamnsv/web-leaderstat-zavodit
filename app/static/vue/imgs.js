'use_strict';
var Imgs = {

    props: ['name', 'imgs'], 
	

    template:   `
	<div class="imgs">
			<ul class="imgs__lst mb-2" :style="'width: '+180*imgs.length+'px'">
					<li class="imgs__it" v-for="(v, k) in imgs">
						<img :src="'dataset/'+name+'/'+v" 
						class="img-thumbnail" :alt="name+v">
					</li>
				</ul>
			</div>
	`
};//Imgs  


