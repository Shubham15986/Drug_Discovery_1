// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Enhance 3D viewer functionality
    enhance3DViewer();
    
    // Add molecule animation
    addMoleculeAnimation();
    
    // Add measurement tooltips
    enhanceMeasurements();
    
    // Improve style controls
    enhanceStyleControls();
});

// Enhance 3D viewer functionality
function enhance3DViewer() {
    // Add loading animation to viewers
    const viewers = ['viewer', 'viewer3D'];
    viewers.forEach(viewerId => {
        const viewer = document.getElementById(viewerId);
        if (!viewer) return;
        
        // Add loading class when loading starts
        const originalView3DStructure = window.view3DStructure;
        if (originalView3DStructure && viewerId === 'viewer') {
            window.view3DStructure = function(sdfUrl) {
                if (viewer) viewer.classList.add('loading');
                originalView3DStructure.apply(this, arguments);
            };
        }
        
        const originalView3DBySmiles = window.view3DBySmiles;
        if (originalView3DBySmiles && viewerId === 'viewer3D') {
            window.view3DBySmiles = function() {
                if (viewer) viewer.classList.add('loading');
                originalView3DBySmiles.apply(this, arguments);
            };
        }
    });
    
    // Enhance viewer controls
    const styleControls = document.getElementById('styleControls');
    if (styleControls) {
        // Group controls by category
        const controlGroups = {
            'Molecule Style': ['ball+stick', 'spacefill', 'licorice', 'hyperball', 'line', 'stick'],
            'Surface Options': ['vdw', 'sas', 'ses', 'Remove Surface'],
            'Measurements': ['Distance', 'Angle', 'Dihedral'],
            'View Controls': ['Labels', 'Zoom In', 'Zoom Out', 'Spin', 'Background']
        };
        
        // Clear existing controls
        styleControls.innerHTML = '';
        
        // Create grouped controls
        for (const [groupName, controls] of Object.entries(controlGroups)) {
            const group = document.createElement('div');
            group.className = 'control-group';
            
            const groupTitle = document.createElement('h5');
            groupTitle.textContent = groupName;
            group.appendChild(groupTitle);
            
            const buttonsContainer = document.createElement('div');
            buttonsContainer.className = 'buttons-container';
            
            controls.forEach(control => {
                const button = document.createElement('button');
                button.className = 'style-btn';
                button.textContent = control;
                
                // Set onclick based on control type
                if (groupName === 'Molecule Style') {
                    button.onclick = function() { window.setNGLStyle(control.toLowerCase()); };
                } else if (groupName === 'Surface Options') {
                    if (control === 'Remove Surface') {
                        button.onclick = function() { window.removeSurface(); };
                    } else {
                        button.onclick = function() { window.addSurface(control.toLowerCase()); };
                    }
                } else if (groupName === 'Measurements') {
                    button.onclick = function() { window['measureNGL' + control](); };
                } else if (groupName === 'View Controls') {
                    if (control === 'Labels') {
                        button.onclick = function() { window.toggleLabels(); };
                    } else if (control === 'Zoom In') {
                        button.onclick = function() { window.zoomIn(); };
                    } else if (control === 'Zoom Out') {
                        button.onclick = function() { window.zoomOut(); };
                    } else if (control === 'Spin') {
                        button.onclick = function() { window.toggleSpin(); };
                    } else if (control === 'Background') {
                        button.onclick = function() { window.toggleBackground(); };
                    }
                }
                
                buttonsContainer.appendChild(button);
            });
            
            group.appendChild(buttonsContainer);
            styleControls.appendChild(group);
        }
    }
}

// Add molecule animation to the page
function addMoleculeAnimation() {
    // Create molecule animation container
    const moleculeAnimation = document.createElement('div');
    moleculeAnimation.className = 'molecule-animation';
    moleculeAnimation.innerHTML = `
        <div class="atom atom-1"></div>
        <div class="atom atom-2"></div>
        <div class="atom atom-3"></div>
        <div class="bond bond-1"></div>
        <div class="bond bond-2"></div>
    `;
    
    // Insert after the h1 element
    const h1 = document.querySelector('h1');
    if (h1 && h1.parentNode) {
        h1.parentNode.insertBefore(moleculeAnimation, h1.nextSibling);
    }
}

// Enhance measurement functionality
function enhanceMeasurements() {
    const originalMeasureDistance = window.measureNGLDistance;
    if (originalMeasureDistance) {
        window.measureNGLDistance = function() {
            // Create tooltip for instructions
            const viewer = document.getElementById('viewer');
            if (viewer) {
                const tooltip = document.createElement('div');
                tooltip.className = 'measurement-tooltip';
                tooltip.textContent = 'Click two atoms to measure distance';
                viewer.appendChild(tooltip);
                
                // Remove tooltip after 5 seconds
                setTimeout(() => {
                    if (tooltip.parentNode) {
                        tooltip.parentNode.removeChild(tooltip);
                    }
                }, 5000);
            }
            
            originalMeasureDistance.apply(this, arguments);
        };
    }
    
    const originalMeasureAngle = window.measureNGLAngle;
    if (originalMeasureAngle) {
        window.measureNGLAngle = function() {
            // Create tooltip for instructions
            const viewer = document.getElementById('viewer');
            if (viewer) {
                const tooltip = document.createElement('div');
                tooltip.className = 'measurement-tooltip';
                tooltip.textContent = 'Click three atoms to measure angle';
                viewer.appendChild(tooltip);
                
                // Remove tooltip after 5 seconds
                setTimeout(() => {
                    if (tooltip.parentNode) {
                        tooltip.parentNode.removeChild(tooltip);
                    }
                }, 5000);
            }
            
            originalMeasureAngle.apply(this, arguments);
        };
    }
}

// Enhance style controls
function enhanceStyleControls() {
    // Add animation to style buttons
    document.querySelectorAll('.style-btn').forEach(button => {
        button.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
            this.style.boxShadow = '0 2px 5px rgba(0, 0, 0, 0.1)';
        });
        
        button.addEventListener('mouseleave', function() {
            this.style.transform = '';
            this.style.boxShadow = '';
        });
        
        // Add active state
        button.addEventListener('click', function() {
            // Remove active class from all buttons in the same group
            const buttonsContainer = this.parentNode;
            if (buttonsContainer) {
                buttonsContainer.querySelectorAll('.style-btn').forEach(btn => {
                    btn.classList.remove('active');
                });
            }
            
            // Add active class to clicked button
            this.classList.add('active');
        });
    });
}