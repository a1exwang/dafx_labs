# DAFX experiments

Reading the book [Digital Audio Effects(DAFX)](https://www.amazon.com/DAFX-Digital-Effects-Udo-Z%C3%B6lzer/dp/0470665998), and implementing the synthesizers, filters and effects.

# Goals

- *Correctly* implement all the effects in the book and other common effect that are not present in the book but a *reliable implementation* can be found, using its original implementation or other simpler and faster implementations.
- *Visualize* as much as possible to get a better understanding of the underlying algorithm. 
- *Simple and Easy to Understand*, we use Python to make the code simpler and easier to understand.
- We can *interactively* changing the parameters and see what happens. To achieve it, we use Jupyter as our UI.

# Non-goals

- This is not a project for production use, so performance is not our main goal, as long as it does not affect playability.
- Real time rendering is not our goal, either.


# Done

- Basic stuff
	- Channels and connections(like in any DAW)
	- Complex number intermediates
	- Impulse response plotting 
	- Spectrum plotting
- Synthesizers
	- Simple oscillators(Sine, Saw, Noise)
	- Sampler
- Filters and Delays
	- Comb filter(delay line), GrossBeat
	- IIR filter
- Modulators
	- Single Side Band
	- Ring Modulator
	- Amplitude Modulator
	- Phase Modulator
	- Frequence Modulator
- Non-linear Processing
	- Limiter
	- Compressor
	- Expander


